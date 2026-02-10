import torch

class RadixNode:
    def __init__(self):
        self.children = {}
        self.tokens = []
        self.kv_cache = None

class RadixCache:
    def __init__(self):
        self.root = RadixNode()
    
    def match_prefix(self, tokens):
        node = self.root
        matched_len = 0
        tokens_list = tokens.tolist() if torch.is_tensor(tokens) else tokens

        while True:
            if matched_len < len(tokens_list) and tokens_list[matched_len] in node.children:
                child = node.children[tokens_list[matched_len]]
                # Match tokens from the child node
                common = 0
                for i, t in enumerate(child.tokens):
                    if matched_len + i < len(tokens_list) and tokens_list[matched_len + i] == t:
                        common += 1
                    else:
                        break
            
                if common == len(child.tokens):
                    matched_len += common
                    node = child # Case of complete match
                else:
                    # Case of partial match
                    full_match_len = matched_len + common
                    # Take the portion of the cache for the common length
                    sliced_cache = []
                    for layer_kv in child.kv_cache:
                        k, v = layer_kv
                        k_slice = k[..., :full_match_len, :]
                        v_slice = v[..., :full_match_len, :]
                        sliced_cache.append((k_slice, v_slice))
                    return sliced_cache, full_match_len
            else:
                break

        return node.kv_cache, matched_len
    
    def insert(self, tokens, kv_cache):
        tokens = tokens.tolist() if torch.is_tensor(tokens) else tokens
        node = self.root
        index = 0

        while index < len(tokens):
            current = tokens[index]
            if current not in node.children:
                # No matching child is found
                new_node = RadixNode()
                new_node.tokens = tokens[index:]
                new_node.kv_cache = kv_cache
                node.children[current] = new_node
                return
            
            # Found the child that shares the first token
            child = node.children[current]
            # Calculate the length of the common prefix
            common = 0
            while (common < len(child.tokens)) and index + common < len(tokens) and child.tokens[common] == tokens[index + common]:
                common += 1
            
            if common == len(child.tokens):
                # Case of full match
                node = child
                index += common

                # If input sequence also ends here
                if index == len(tokens):
                    node.kv_cache = kv_cache
                    return
            else:
                # Partial match case
                split_node = RadixNode()
                split_node.tokens = child.tokens[:common]
                split_node.kv_cache = self._slice_kv_cache(kv_cache, index + common)
                split_node.children = {}
                # Fix the Existing Child
                remaining_child_tokens = child.tokens[common:]
                child.tokens = remaining_child_tokens
                split_node.children[remaining_child_tokens[0]] = child
                # Creating the new suffix
                remaining_new_tokens = tokens[index + common:]
                if remaining_child_tokens:
                    new_leaf = RadixNode()
                    new_leaf.tokens = remaining_new_tokens
                    new_leaf.kv_cache = kv_cache
                    split_node.children[remaining_new_tokens[0]] = new_leaf
                else:
                    pass

                node.children[current] = split_node
                return
    
    def _slice_kv_cache(self, kv_cache, length):
        if kv_cache is None: return None

        sliced_cache = []
        for k, v in kv_cache:
            k_slice = k[..., :length, :].clone()
            v_slice = v[..., :length, :].clone()
            sliced_cache.append((k_slice, v_slice))

        return sliced_cache