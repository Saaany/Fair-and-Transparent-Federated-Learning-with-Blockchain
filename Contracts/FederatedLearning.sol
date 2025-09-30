// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FederatedLearning {
    struct Node {
        address addr;
        uint256 tokens;
    }

    mapping(address => Node) public nodes;
    mapping(address => bool) public registered;

    event ModelSubmitted(address node, string modelHash);

    constructor() {}

    // Register a new node
    function registerNode() public {
        require(!registered[msg.sender], "Already registered");
        nodes[msg.sender] = Node(msg.sender, 0);
        registered[msg.sender] = true;
    }

    // Submit model update hash (SHA256 string)
    function submitUpdate(string memory modelHash) public {
        require(registered[msg.sender], "Not registered");
        emit ModelSubmitted(msg.sender, modelHash);
        nodes[msg.sender].tokens += 10;  // Reward tokens
    }

    // Check tokens
    function getTokens(address user) public view returns (uint256) {
        return nodes[user].tokens;
    }
}
