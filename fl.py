import hashlib

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from web3 import Web3

# -----------------------------
# Blockchain Setup
# -----------------------------
GANACHE_URL = "http://127.0.0.1:7545"
w3 = Web3(Web3.HTTPProvider(GANACHE_URL))

client_accounts = w3.eth.accounts[2:8]  # 6 clients

contract_address = "0xC1eEBa3bB6E24B64ebA088CEDE7BdD15aE959d73"
abi = [
	{
		"inputs": [],
		"stateMutability": "nonpayable",
		"type": "constructor"
	},
	{
		"anonymous": False,
		"inputs": [
			{
				"indexed": False,
				"internalType": "address",
				"name": "node",
				"type": "address"
			},
			{
				"indexed": False,
				"internalType": "string",
				"name": "modelHash",
				"type": "string"
			}
		],
		"name": "ModelSubmitted",
		"type": "event"
	},
	{
		"inputs": [
			{
				"internalType": "address",
				"name": "user",
				"type": "address"
			}
		],
		"name": "getTokens",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "address",
				"name": "",
				"type": "address"
			}
		],
		"name": "nodes",
		"outputs": [
			{
				"internalType": "address",
				"name": "addr",
				"type": "address"
			},
			{
				"internalType": "uint256",
				"name": "tokens",
				"type": "uint256"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "registerNode",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "address",
				"name": "",
				"type": "address"
			}
		],
		"name": "registered",
		"outputs": [
			{
				"internalType": "bool",
				"name": "",
				"type": "bool"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "modelHash",
				"type": "string"
			}
		],
		"name": "submitUpdate",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	}
]  
contract = w3.eth.contract(address=contract_address, abi=abi)

# Register clients (skip if already registered)
print("Checking/ registering clients...")
for acct in client_accounts[0:3]:
    try:
        # Try to get tokens → if works, already registered
        _ = contract.functions.getTokens(acct).call()
        print(f"✔ Client {acct} already registered")
    except:
        # If fails, register
        tx_hash = contract.functions.registerNode().transact({'from': acct})
        w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"✔ Client {acct} newly registered")

for acct in client_accounts[3:6]:
	# try:
	# 	# Try to get tokens → if works, already registered
	# 	_ = contract.functions.getTokens(acct).call()
	# 	print(f"✔ Client {acct} already registered")
	# except:
	# If fails, register
	tx_hash = contract.functions.registerNode().transact({'from': acct})
	w3.eth.wait_for_transaction_receipt(tx_hash)
	print(f"✔ Client {acct} newly registered")	

# -----------------------------
# Model Definition
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# Federated Setup
# -----------------------------
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split dataset into 6 equal parts
client_data = random_split(dataset, [10000, 10000, 10000, 10000, 10000, 10000])
clients = [DataLoader(cd, batch_size=64, shuffle=True) for cd in client_data]

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

def train_local(model, train_loader, epochs=1):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def test(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(test_loader.dataset)

# -----------------------------
# Federated Learning + Blockchain
# -----------------------------
global_model = SimpleCNN()
rounds = 15
accuracies = []
gas_usage = []

for r in range(rounds):
    local_weights = []
    print(f"\n=== Round {r+1} ===")

    for i, loader in enumerate(clients):
        local_model = SimpleCNN()
        local_model.load_state_dict(global_model.state_dict())
        state_dict = train_local(local_model, loader)

        # Hash update
        model_bytes = str(state_dict).encode()
        update_hash = hashlib.sha256(model_bytes).hexdigest()
        print(f"Client {i+1} hash: {update_hash[:10]}...")

        # Submit update
        tx_hash = contract.functions.submitUpdate(update_hash).transact({'from': client_accounts[i]})
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        gas_used = receipt.gasUsed
        gas_usage.append({"Round": r+1, "Client": i+1, "GasUsed": gas_used})
        print(f"✔ Client {i+1} submitted (gas used: {gas_used})")

        local_weights.append(state_dict)

    # FedAvg aggregation
    new_state_dict = global_model.state_dict()
    for key in new_state_dict.keys():
        new_state_dict[key] = sum(w[key] for w in local_weights) / len(local_weights)
    global_model.load_state_dict(new_state_dict)

    # Test accuracy
    acc = test(global_model)
    accuracies.append({"Round": r+1, "Accuracy": acc})
    print(f"Global Accuracy after round {r+1}: {acc:.4f}")

# -----------------------------
# Token Balances
# -----------------------------
balances = []
print("\nFinal Token Balances:")
for i, acct in enumerate(client_accounts):
    tokens = contract.functions.getTokens(acct).call()
    balances.append({"Client": i+1, "Account": acct, "Tokens": tokens})
    print(f"Client {i+1}: {tokens} tokens")

# -----------------------------
# Save to CSV/Excel
# -----------------------------
df_acc = pd.DataFrame(accuracies)
df_gas = pd.DataFrame(gas_usage)
df_bal = pd.DataFrame(balances)

df_acc.to_csv("accuracy.csv", index=False)
df_gas.to_csv("gas_usage.csv", index=False)
df_bal.to_csv("balances.csv", index=False)

with pd.ExcelWriter("federated_results.xlsx") as writer:
    df_acc.to_excel(writer, sheet_name="Accuracy", index=False)
    df_gas.to_excel(writer, sheet_name="GasUsage", index=False)
    df_bal.to_excel(writer, sheet_name="Balances", index=False)

print("\n✔ Results saved: accuracy.csv, gas_usage.csv, balances.csv, federated_results.xlsx")

# -----------------------------
# Plot Accuracy
# -----------------------------
plt.plot(df_acc["Round"], df_acc["Accuracy"], marker='o')
plt.xlabel("Federated Rounds")
plt.ylabel("Test Accuracy")
plt.title("Federated Learning Accuracy")
plt.grid()
plt.show()
plt.show()
plt.show()
