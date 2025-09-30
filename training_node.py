import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import hashlib
from web3 import Web3

# ==== Blockchain Setup ====
ganache_url = "http://127.0.0.1:7545"
w3 = Web3(Web3.HTTPProvider(ganache_url))

acct = w3.eth.accounts[0]  # Use first Ganache account

# Replace these with your contract details from Remix
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
]  # Paste ABI JSON here

contract = w3.eth.contract(address=contract_address, abi=abi)

# ==== Federated Learning Simulation ====
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Simple model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        return self.fc(x.view(-1, 28*28))

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Train for 1 batch (demo only)
for images, labels in trainloader:
    optimizer.zero_grad()
    outputs = net(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    break

# Hash model weights
weights = torch.cat([p.view(-1) for p in net.parameters()])
weights_hash = hashlib.sha256(weights.detach().numpy().tobytes()).hexdigest()
print("Model Hash:", weights_hash)

# ==== Interact with Smart Contract ====
# Register node (only once per account)
tx = contract.functions.registerNode().transact({'from': acct})
w3.eth.wait_for_transaction_receipt(tx)
print("Node registered on blockchain.")

# Submit model update
tx = contract.functions.submitUpdate(weights_hash).transact({'from': acct})
w3.eth.wait_for_transaction_receipt(tx)
print("Model update submitted!")

# Check tokens
tokens = contract.functions.getTokens(acct).call()
print("Tokens earned:", tokens)
