use std::fs::{File, OpenOptions};
use std::io::{self, Write, Seek, SeekFrom};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use memmap2::{MmapMut, MmapOptions};
use std::cmp::Ordering;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

const HEADER: usize = 4;
const BTREE_PAGE_SIZE: usize = 4096;
const BTREE_MAX_KEY_SIZE: usize = 1000;
const BTREE_MAX_VAL_SIZE: usize = 3000;
const BNODE_NODE: u16 = 1; // internal nodes without values
const BNODE_LEAF: u16 = 2; // leaf nodes with values
const BNODE_FREE: u16 = 3; // free nodes
const FREE_LIST_HEADER: usize = 4 + 8 + 8;
const FREE_LIST_CAP: usize = (BTREE_PAGE_SIZE - FREE_LIST_HEADER) / 8;






const DB_SIG: &[u8] = b"BuildYourOwnDB05";
use tempfile::NamedTempFile;



//* Verify that a single KV pair fits on a page

const _: () = {
    let node1max = HEADER + 8 + 2 + 4 + BTREE_MAX_KEY_SIZE + BTREE_MAX_VAL_SIZE;
    assert!(node1max <= BTREE_PAGE_SIZE);
};

#[derive(Clone)]
pub struct BNode {
    data : Vec<u8>,
}

pub struct BTree {
    root : u64,
    get :Box<dyn Fn(u64) -> BNode>,
    new : Box<dyn Fn(BNode) -> u64>,
    del : Box<dyn Fn(u64)>,
}

impl BNode {
    pub fn new(size : usize) -> Self {
        Self {
            data : vec![0; size],
        }
    }

    pub fn from_bytes(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(), // ou data.into() si tu veux vraiment copier
        }
    }

    // * Header accessors
    pub fn btype(&self) -> u16 {
        u16::from_le_bytes([self.data[0], self.data[1]])
    }
    
    pub fn nkeys(&self) -> u16 {
        u16::from_le_bytes([self.data[2], self.data[3]])
    }

    pub fn setHeader(&mut self, btype : u16, nkeys : u16) {
        self.data[0..2].copy_from_slice(&btype.to_le_bytes());
        self.data[2..4].copy_from_slice(&nkeys.to_le_bytes());
    }

    pub fn getPtr(&self, index : u16) -> u64 {
        assert!(index < self.nkeys());
        let pos = HEADER + index as usize * 8;
        u64::from_le_bytes([
            self.data[pos], self.data[pos + 1], self.data[pos + 2], self.data[pos + 3],
            self.data[pos + 4], self.data[pos + 5], self.data[pos + 6], self.data[pos + 7],
        ])
    }

    pub fn setPtr(&mut self, index : u16, val : u64) {
        assert!(index < self.nkeys());
        let pos = HEADER + index as usize * 8;
        self.data[pos..pos + 8].copy_from_slice(&val.to_le_bytes());
    }

    pub fn offsetPos(&self, index : u16) -> usize {
        assert!(index >= 1 && index <= self.nkeys());
        HEADER + 8 * self.nkeys() as usize + 2 * (index - 1) as usize
    }

    pub fn getOffset(&self, index : u16) -> u16 {
        if index == 0 {
            return 0;
        }
        
        assert!(index <= self.nkeys());
        let pos = self.offsetPos(index);
        u16::from_le_bytes([self.data[pos], self.data[pos + 1]])
    }

    pub fn setOffset(&mut self, index : u16, offset : u16) {
        if index == 0 {
            return;
        }
        
        assert!(index <= self.nkeys());
        let pos = self.offsetPos(index);
        self.data[pos..pos + 2].copy_from_slice(&offset.to_le_bytes());
    }

    // Key-Value accesors
    pub fn kvPos(&self, index : u16) -> u16 {
        assert!(index <= self.nkeys());
        (HEADER + 8 * self.nkeys() as usize + 2 * self.nkeys() as usize) as u16 + self.getOffset(index)
    }

    pub fn getKey(&self, index : u16) -> &[u8] {
        assert!(index < self.nkeys());
        let pos = self.kvPos(index) as usize;
        let klen = u16::from_le_bytes([self.data[pos], self.data[pos + 1]]) as usize;
        &self.data[pos + 4..pos + 4 + klen]
    }

    pub fn getVal(&self, index : u16) -> &[u8] {
        assert!(index < self.nkeys());
        let pos = self.kvPos(index) as usize;
        let klen = u16::from_le_bytes([self.data[pos], self.data[pos + 1]]) as usize;
        let vlen = u16::from_le_bytes([self.data[pos + 2], self.data[pos + 3]]) as usize;
        &self.data[pos + 4 + klen..pos + 4 + klen + vlen]
    }
    pub fn nbytes(&self) -> u16 {
        self.kvPos(self.nkeys())
    }
}
// * Step 1 : Look up the key
// * Returns the first kid node whose range intersects the key (kid[i] <= key)
fn nodeLookupLe(node : &BNode, key: &[u8]) -> u16 {
    let nkeys = node.nkeys();
    let mut found = 0;

    // The first key is a copy from the parent node,
    // thus it's always less than or equal to the key.
    for i in 1..nkeys {
        match node.getKey(i).cmp(key) {
            Ordering::Less | Ordering::Equal => found = i,
            Ordering::Greater => break,
        }
    }
    found
}

// * Step 2 : Update Leaf Node
// * Add a new key to a leaf node
fn leafInsert(new : &mut BNode, old: &BNode, index: u16, key: &[u8], val: &[u8]) {
    new.setHeader(BNODE_LEAF, old.nkeys() + 1);
    
    // Copy key-value pairs before the insertion point
    nodeAppendRange(new, old, 0, 0, index);
    
    // Insert the new key-value pair
    nodeAppendKv(new, index, 0, key, val);
    
    // Copy the remaining key-value pairs, adjusting their positions
    nodeAppendRange(new, old, index + 1, index, old.nkeys() - index);
}

fn leafUpdate(new : &mut BNode, old : &BNode, index : u16, key : &[u8], val : &[u8]) {
    new.setHeader(BNODE_LEAF, old.nkeys());
    
    // Copy the key-value pairs up to the one we're updating
    nodeAppendRange(new, old, 0, 0, index);
    
    // Insert the updated key-value pair
    nodeAppendKv(new, index, old.getPtr(index), key, val);
    
    // Copy the remaining key-value pairs
    nodeAppendRange(new, old, index + 1, index + 1, old.nkeys() - index - 1);
}


// * Append a range of nodes from one node to another
fn nodeAppendRange(new : &mut BNode, old : &BNode, dstNew : u16 , srcOld : u16 , n : u16 ) {
    assert!(srcOld + n <= old.nkeys());
    assert!(dstNew + n <= new.nkeys());

    if n == 0 {
        return;
    }

    // * Copy the pointers
    for i in 0..n {
        new.setPtr(dstNew + i, old.getPtr(srcOld +i));
    }

    // * Copy the offsets
    let dstBegin = new.getOffset(dstNew);
    let srcBegin = old.getOffset(srcOld);
    for i in 1..=n {
        let offset = dstBegin + old.getOffset(srcOld + i) - srcBegin;
        new.setOffset(dstNew + i, offset);
    }

    //* Copy the KV
    let begin = old.kvPos(srcOld) as usize;
    let end = old.kvPos(srcOld + n) as usize;
    let dstPos = new.kvPos(dstNew) as usize;
    new.data[dstPos..dstPos + (end - begin)].copy_from_slice(&old.data[begin..end]);
}


// * Copy a KV pair to position
fn nodeAppendKv(new : &mut BNode , index : u16, ptr : u64, key : &[u8], val : &[u8]) {
    
    //* Set the pointer
    new.setPtr(index, ptr);

    //* Set KV Data
    let pos = new.kvPos(index) as usize;
    new.data[pos..pos + 2].copy_from_slice(&(key.len() as u16).to_le_bytes());
    new.data[pos + 2..pos + 4].copy_from_slice(&(val.len() as u16).to_le_bytes());
    new.data[pos + 4..pos + 4 + key.len()].copy_from_slice(key);
    new.data[pos + 4 + key.len()..pos + 4 + key.len() + val.len()].copy_from_slice(val);

    //* Set the offset
    let nextOffset = new.getOffset(index) + 4 + key.len() as u16 + val.len() as u16;
    new.setOffset(index + 1, nextOffset);
}

// * Insert the kv into a node , the result might be split into two nodes
fn treeInsert(tree : &mut BTree, node : BNode , key : &[u8], val: &[u8]) -> BNode {
    let mut new = BNode::new(2 * BTREE_PAGE_SIZE);
    let index = nodeLookupLe(&node , key);
    println!("treeInsert: node.btype() = {}", node.btype());

    match node.btype() {
        BNODE_LEAF => {
            // Leaf node, node.getKey(index) <= key
            if index < node.nkeys() && key == node.getKey(index) {
                // Found the key, update it
                leafUpdate(&mut new, &node, index, key, val);
            } else {
                // Insert it after the position
                leafInsert(&mut new, &node, index + 1, key, val);
            }
        }
        BNODE_NODE => {
            // Internal node, insert it to a kid node
            nodeInsert(tree, &mut new, &node, index, key, val);
        }
        _ => panic!("Invalid node type"),
    }
    new
}

//* Step 4: Handle Internal Nodes
//* Part of the tree_insert(): KV insertion to an internal node
fn nodeInsert(
    tree : &mut BTree,
    new : &mut BNode,
    old : &BNode,
    index : u16,
    key : &[u8],
    val : &[u8],
) {
    let kptr = old.getPtr(index);
    let knode = (tree.get)(kptr);
    (tree.del)(kptr);

    let knode = treeInsert(tree, knode, key, val);
    let (nsplit, splitnodes) = nodeSplit3(knode);

    nodeReplaceKidN(tree, new , old , index , &splitnodes[..nsplit as usize]);
}

// * Delete from a leaf node

fn leafDelete(new : &mut BNode, old : &BNode, index : u16) {
    let newnkeys = old.nkeys() - 1;
    if newnkeys == 0 {
        return;
    }
    new.setHeader(BNODE_LEAF, newnkeys);
    nodeAppendRange(new, old, 0, 0, index);
    nodeAppendRange(new, old, index, index + 1, old.nkeys() - index - 1);
}

fn nodeDeleteUpdate(new : &mut BNode , old : &BNode , index : u16 , updatedchild : BNode , tree : &mut BTree) {
    new.setHeader(BNODE_NODE, old.nkeys());
    nodeAppendRange(new, old, 0, 0, index);
    let childptr = (tree.new)(updatedchild.clone());
    nodeAppendKv(new, index, childptr, updatedchild.getKey(0), &[]);
    nodeAppendRange(new, old, index + 1, index + 1, old.nkeys() - index - 1);
}

// * Delete a key from the tree

fn treeDelete(tree : &mut BTree, old : BNode, key : &[u8]) -> BNode {
    let index = nodeLookupLe(&old, key);
    match old.btype() {
        BNODE_LEAF => {
            if index >= old.nkeys() || old.getKey(index) != key {
                return old;
            }
            let mut new = BNode::new(BTREE_PAGE_SIZE);
            leafDelete(&mut new, &old, index);
            new
        }
        BNODE_NODE => {
            let childptr = old.getPtr(index);
            let childnode = (tree.get)(childptr);
            (tree.del)(childptr);

            let updatedchild = treeDelete(tree, childnode, key);

            if updatedchild.nkeys() == 0 {
                return BNode::new(0);
            }

            let mut new = BNode::new(BTREE_PAGE_SIZE);
            nodeDeleteUpdate(&mut new, &old, index, updatedchild, tree);
            new
        }
        _ => panic!("Invalid node type"),
    }
}




//* Step 5 : Split big node

fn nodeSplit2(left: &mut BNode , right : &mut BNode , old : &BNode) {
    
    let nkeys = old.nkeys();
    let splitidx = nkeys / 2;

    left.setHeader(old.btype(), splitidx);
    nodeAppendRange(left, old, 0, 0, splitidx);

    let rightnkeys = nkeys - splitidx;
    right.setHeader(old.btype(), rightnkeys);
    nodeAppendRange(right, old, 0, splitidx, rightnkeys);
}

fn nodeSplit3(old : BNode) -> (u16, [BNode;3]) {
    
    if old.nbytes() as usize <= BTREE_PAGE_SIZE {
        let mut trimmed = old;
        trimmed.data.truncate(BTREE_PAGE_SIZE);
        return (1, [trimmed, BNode::new(0), BNode::new(0)]);
    }

    let mut left = BNode::new(2 * BTREE_PAGE_SIZE);
    let mut right = BNode::new(BTREE_PAGE_SIZE);
    nodeSplit2(&mut left, &mut right, &old);
    
    if left.nbytes() as usize <= BTREE_PAGE_SIZE {
        left.data.truncate(BTREE_PAGE_SIZE);
        return (2, [left, right, BNode::new(0)]);
    }

    let mut leftleft = BNode::new(BTREE_PAGE_SIZE);
    let mut middle = BNode::new(BTREE_PAGE_SIZE);
    nodeSplit2(&mut leftleft, &mut middle, &left);
    
    assert!(leftleft.nbytes() as usize <= BTREE_PAGE_SIZE);
    (3, [leftleft, middle, right])
}

//* Step 6 : Update internal node

fn nodeReplaceKidN(
    tree: &mut BTree,
    new: &mut BNode,
    old: &BNode,
    index: u16,
    kids: &[BNode],
) {
    let inc = kids.len() as u16;
    let newnkeys = old.nkeys() + inc - 1;
    new.setHeader(BNODE_NODE, newnkeys);
    nodeAppendRange(new, old, 0, 0, index);

    for (i,kid) in kids.iter().enumerate() {
        let kidptr = (tree.new)(kid.clone());
        nodeAppendKv(new, index + i as u16, kidptr, kid.getKey(0), &[]);
    }

    nodeAppendRange(new, old, index + inc, index + 1, old.nkeys() - index - 1);
}

impl BTree {
    pub fn new() -> Self {
        Self {
            root: 0,
            get: Box::new(|_| BNode::new(BTREE_PAGE_SIZE)),
            new: Box::new(|_| 1),
            del: Box::new(|_| {}),
        }
    }
    
    pub fn insert(&mut self, key: &[u8], val: &[u8]) {
        assert!(!key.is_empty());
        assert!(key.len() <= BTREE_MAX_KEY_SIZE);
        assert!(val.len() <= BTREE_MAX_VAL_SIZE);
        
        if self.root == 0 {
            // Create root node with a dummy key
            let mut root = BNode::new(BTREE_PAGE_SIZE);
            root.setHeader(BNODE_LEAF, 2); // 2 keys: dummy + actual
            
            // Add a dummy key (empty) that covers the whole key space
            // This ensures nodeLookupLe always finds a containing node
            nodeAppendKv(&mut root, 0, 0, &[], &[]);
            
            // Add the actual key
            nodeAppendKv(&mut root, 1, 0, key, val);
            
            self.root = (self.new)(root);
            return;
        }
        
        let root_node = (self.get)(self.root);
        (self.del)(self.root);
        let new_root = treeInsert(self, root_node, key, val);
        
        // Handle root splitting
        let (nsplit, split_nodes) = nodeSplit3(new_root);
        if nsplit > 1 {
            // Create new root
            let mut root = BNode::new(BTREE_PAGE_SIZE);
            root.setHeader(BNODE_NODE, nsplit);
            for (i, node) in split_nodes[..nsplit as usize].iter().enumerate() {
                let ptr = (self.new)(node.clone());
                nodeAppendKv(&mut root, i as u16, ptr, node.getKey(0), &[]);
            }
            self.root = (self.new)(root);
        } else {
            self.root = (self.new)(split_nodes[0].clone());
        }
    }
    
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        if self.root == 0 {
            return None;
        }
        
        let mut node = (self.get)(self.root);
        
        loop {
            let idx = nodeLookupLe(&node, key);
            println!("treeInsert: node.btype() = {}", node.btype());

            match node.btype() {
                BNODE_LEAF => {
                    // For leaf nodes, we need to check if we found the exact key
                    if idx < node.nkeys() && node.getKey(idx) == key {
                        return Some(node.getVal(idx).to_vec());
                    } else {
                        return None;
                    }
                }
                BNODE_NODE => {
                    // For internal nodes, we follow the pointer
                    let ptr = node.getPtr(idx);
                    node = (self.get)(ptr);
                }
                _ => panic!("Bad node type!"),
            }
        }
    }

    pub fn delete(&mut self, key : &[u8]) -> bool {
        assert!(!key.is_empty());
        assert!(key.len() <=  BTREE_MAX_KEY_SIZE);

        if self.root == 0 {
            return false;
        }

        let rootnode = (self.get)(self.root);
        let updated = treeDelete(self, rootnode, key);

        if updated.nkeys() == 0 {
            return false;
        }

        (self.del)(self.root);

        if updated.btype() == BNODE_NODE && updated.nkeys() == 1 {
            self.root = updated.getPtr(0);
        } else {
            self.root = (self.new)(updated);
        }
        
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use tempfile::NamedTempFile;

    // Mock page manager for testing
    struct MockPageManager {
        pages: Arc<Mutex<HashMap<u64, BNode>>>,
        next_id: Arc<Mutex<u64>>,
    }

    impl MockPageManager {
        fn new() -> Self {
            Self {
                pages: Arc::new(Mutex::new(HashMap::new())),
                next_id: Arc::new(Mutex::new(1)),
            }
        }

        fn create_btree(&self) -> BTree {
            let pages_get = Arc::clone(&self.pages);
            let pages_new = Arc::clone(&self.pages);
            let pages_del = Arc::clone(&self.pages);
            let next_id_new = Arc::clone(&self.next_id);

            BTree {
                root: 0,
                get: Box::new(move |id| {
                    pages_get.lock().unwrap().get(&id).unwrap().clone()
                }),
                new: Box::new(move |node| {
                    let mut next_id = next_id_new.lock().unwrap();
                    let id = *next_id;
                    *next_id += 1;
                    pages_new.lock().unwrap().insert(id, node);
                    id
                }),
                del: Box::new(move |id| {
                    pages_del.lock().unwrap().remove(&id);
                }),
            }
        }
    }

    #[test]
    fn test_btree_insert_and_get() {
        let page_manager = MockPageManager::new();
        let mut btree = page_manager.create_btree();
        
        // Insert first key-value pair
        println!("Inserting key1=value1");
        btree.insert(b"key1", b"value1");
        
        // Test first retrieval
        let result1 = btree.get(b"key1");
        println!("get(key1) returned: {:?}", result1);
        assert_eq!(result1, Some(b"value1".to_vec()));
        
        // Insert second key-value pair
        println!("\nInserting key2=value2");
        btree.insert(b"key2", b"value2");
        
        // Test second retrieval
        let result2 = btree.get(b"key2");
        println!("get(key2) returned: {:?}", result2);
        assert_eq!(result2, Some(b"value2".to_vec()));
        
        // Insert third key-value pair
        println!("\nInserting key3=value3");
        btree.insert(b"key3", b"value3");
        let result3 = btree.get(b"key3");
        println!("get(key3) returned: {:?}", result3);
        assert_eq!(result3, Some(b"value3".to_vec()));
        
        // Check nonexistent key
        assert_eq!(btree.get(b"nonexistent"), None);
    }

    #[test]
    fn test_node_format() {
        let mut node = BNode::new(BTREE_PAGE_SIZE);
        node.setHeader(BNODE_LEAF, 2);
        
        // Test header
        assert_eq!(node.btype(), BNODE_LEAF);
        assert_eq!(node.nkeys(), 2);
        
        // Test KV operations
        nodeAppendKv(&mut node, 0, 0, b"key1", b"value1");
        nodeAppendKv(&mut node, 1, 0, b"key2", b"value2");
        
        assert_eq!(node.getKey(0), b"key1");
        assert_eq!(node.getVal(0), b"value1");
        assert_eq!(node.getKey(1), b"key2");
        assert_eq!(node.getVal(1), b"value2");
    }

    #[test]
    fn test_btree_delete() {
        let page_manager = MockPageManager::new();
        let mut btree = page_manager.create_btree();
        
        // Insert several key-value pairs
        println!("Inserting key1=value1");
        btree.insert(b"key1", b"value1");
        println!("Inserting key2=value2");
        btree.insert(b"key2", b"value2");
        println!("Inserting key3=value3");
        btree.insert(b"key3", b"value3");
        
        // Verify all keys were inserted correctly
        assert_eq!(btree.get(b"key1"), Some(b"value1".to_vec()));
        assert_eq!(btree.get(b"key2"), Some(b"value2".to_vec()));
        assert_eq!(btree.get(b"key3"), Some(b"value3".to_vec()));
        

        // Test 2: Delete a key that exists
        println!("\nDeleting key2");
        let result = btree.delete(b"key2");
        assert!(result, "Deleting existing key should return true");
        
        // Verify key2 is gone but other keys remain
        assert_eq!(btree.get(b"key1"), Some(b"value1".to_vec()));
        assert_eq!(btree.get(b"key2"), None);
        assert_eq!(btree.get(b"key3"), Some(b"value3".to_vec()));
        
        // Test 3: Delete another key
        println!("\nDeleting key1");
        let result = btree.delete(b"key1");
        assert!(result, "Deleting existing key should return true");
        
        // Verify only key3 remains
        assert_eq!(btree.get(b"key1"), None);
        assert_eq!(btree.get(b"key2"), None);
        assert_eq!(btree.get(b"key3"), Some(b"value3".to_vec()));
        
        // Test 4: Delete the last key
        println!("\nDeleting key3 (last key)");
        let result = btree.delete(b"key3");
        assert!(result, "Deleting last key should return true");
        
        // Verify all keys are gone
        assert_eq!(btree.get(b"key1"), None);
        assert_eq!(btree.get(b"key2"), None);
        assert_eq!(btree.get(b"key3"), None);
    }
    
    #[test]
    fn test_btree_complex_delete() {
        let page_manager = MockPageManager::new();
        let mut btree = page_manager.create_btree();
        
        // Insert many keys to create a multi-level tree
        println!("Inserting many keys to create a multi-level tree");
        for i in 0..20 {
            let key = format!("key{:02}", i);
            let val = format!("value{:02}", i);
            btree.insert(key.as_bytes(), val.as_bytes());
        }
        
        // Verify all keys are present
        for i in 0..20 {
            let key = format!("key{:02}", i);
            let val = format!("value{:02}", i);
            assert_eq!(btree.get(key.as_bytes()), Some(val.as_bytes().to_vec()));
        }
        
        // Delete keys in a specific order to test different cases
        let delete_order = [5, 10, 0, 15, 19, 1, 18, 2, 17, 3, 16, 4, 14, 6, 13, 7, 12, 8, 11, 9];
        
        for (order_idx, &i) in delete_order.iter().enumerate() {
            let key = format!("key{:02}", i);
            println!("\nDeleting {}", key);
            let result = btree.delete(key.as_bytes());
            assert!(result, "Deletion should succeed");
            
            // Verify the deleted key is gone
            assert_eq!(btree.get(key.as_bytes()), None);
            
            // Verify remaining keys are still present
            for j in 0..20 {
                if !delete_order[..=order_idx].contains(&j) {
                    let remain_key = format!("key{:02}", j);
                    let remain_val = format!("value{:02}", j);
                    assert_eq!(
                        btree.get(remain_key.as_bytes()), 
                        Some(remain_val.as_bytes().to_vec()),
                        "Key {} should still exist", remain_key
                    );
                }
            }
        }
              // All keys should now be deleted
              for i in 0..20 {
                let key = format!("key{:02}", i);
                assert_eq!(btree.get(key.as_bytes()), None);
            }
        }
    }


pub fn main() {

    fn main() -> std::io::Result<()> {
        // 1. Create a KV instance with a file path
        use std::path::Path;
        let path = Path::new("my_database.db");
        let mut database = KV::new(path);
        
        // 2. Open the database
        database.open()?;
        
        // 3. Store data
        database.set(b"user:1001", b"John Doe")?;
        database.set(b"user:1002", b"Jane Smith")?;
        database.set(b"counter", b"42")?;
        
        // 4. Retrieve data
        if let Some(value) = database.get(b"user:1001") {
            println!("User 1001: {}", String::from_utf8_lossy(&value));
        }
        
        // 5. Delete data
        database.delete(b"counter")?;
        
        // 6. Close the database
        database.close();
        
        Ok(())
    }
}

#[cfg(test)]
mod kv_tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_kv_basic_operations() -> io::Result<()> {
        let temp_file = NamedTempFile::new()?;
        let mut kv = KV::new(temp_file.path());
        
        // Open database
        kv.open()?;
        
        // Test set and get
        kv.set(b"key1", b"value1")?;
        assert_eq!(kv.get(b"key1"), Some(b"value1".to_vec()));
        
        // Test non-existent key
        assert_eq!(kv.get(b"nonexistent"), None);
        
        // Test update existing key
        kv.set(b"key1", b"updated_value1")?;
        assert_eq!(kv.get(b"key1"), Some(b"updated_value1".to_vec()));
        
   
    
        
        kv.close();
        Ok(())
    }
    
    #[test]
    fn test_kv_multiple_keys() -> io::Result<()> {
        let temp_file = NamedTempFile::new()?;
        let mut kv = KV::new(temp_file.path());
        
        kv.open()?;
        
        // Insert multiple keys
        for i in 0..10 {
            let key = format!("key{}", i);
            let value = format!("value{}", i);
            kv.set(key.as_bytes(), value.as_bytes())?;
        }
        
        // Verify all keys
        for i in 0..10 {
            let key = format!("key{}", i);
            let expected_value = format!("value{}", i);
            assert_eq!(kv.get(key.as_bytes()), Some(expected_value.as_bytes().to_vec()));
        }
        
        // Delete even keys
        for i in (0..10).step_by(2) {
            let key = format!("key{}", i);
            assert_eq!(kv.delete(key.as_bytes())?, true);
        }
        
        // Verify deleted and remaining keys
        for i in 0..10 {
            let key = format!("key{}", i);
            let expected_value = format!("value{}", i);
            
            if i % 2 == 0 {
                // Even keys should be deleted
                assert_eq!(kv.get(key.as_bytes()), None);
            } else {
                // Odd keys should remain
                assert_eq!(kv.get(key.as_bytes()), Some(expected_value.as_bytes().to_vec()));
            }
        }
        
        kv.close();
        Ok(())
    }
    
    #[test]
    fn test_kv_persistence() -> io::Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path().to_path_buf(); // Store path for reuse
        
        // First session: Write data
        {
            let mut kv = KV::new(&path);
            kv.open()?;
            
            kv.set(b"persistent_key1", b"persistent_value1")?;
            kv.set(b"persistent_key2", b"persistent_value2")?;
            
            // Verify data was written
            assert_eq!(kv.get(b"persistent_key1"), Some(b"persistent_value1".to_vec()));
            assert_eq!(kv.get(b"persistent_key2"), Some(b"persistent_value2".to_vec()));
            
            kv.close();
        }
        
        // Second session: Read and verify data persists
        {
            let mut kv = KV::new(&path);
            kv.open()?;
            
            // Data should still be accessible
            assert_eq!(kv.get(b"persistent_key1"), Some(b"persistent_value1".to_vec()));
            assert_eq!(kv.get(b"persistent_key2"), Some(b"persistent_value2".to_vec()));
            
            // Update a key
            kv.set(b"persistent_key1", b"updated_value1")?;
            
            kv.close();
        }
        
        // Third session: Verify updates persist
        {
            let mut kv = KV::new(&path);
            kv.open()?;
            
            // Verify update persisted
            assert_eq!(kv.get(b"persistent_key1"), Some(b"updated_value1".to_vec()));
            assert_eq!(kv.get(b"persistent_key2"), Some(b"persistent_value2".to_vec()));
            
            // Delete a key
            assert_eq!(kv.delete(b"persistent_key2")?, true);
            
            kv.close();
        }
        
        // Fourth session: Verify deletion persists
        {
            let mut kv = KV::new(&path);
            kv.open()?;
            
            // First key should still have updated value
            assert_eq!(kv.get(b"persistent_key1"), Some(b"updated_value1".to_vec()));
            
            // Second key should be deleted
            assert_eq!(kv.get(b"persistent_key2"), None);
            
            kv.close();
        }
        
        Ok(())
    }
    
    #[test]
    fn test_kv_large_values() -> io::Result<()> {
        let temp_file = NamedTempFile::new()?;
        let mut kv = KV::new(temp_file.path());
        
        kv.open()?;
        
        // Create a large value (but under BTREE_MAX_VAL_SIZE)
        let large_value = vec![b'x'; 2000];
        
        // Set and get large value
        kv.set(b"large_key", &large_value)?;
        assert_eq!(kv.get(b"large_key"), Some(large_value.clone()));
        
        // Update with a different large value
        let updated_large_value = vec![b'y'; 1500];
        kv.set(b"large_key", &updated_large_value)?;
        assert_eq!(kv.get(b"large_key"), Some(updated_large_value.clone()));
        
        kv.close();
        Ok(())
    }
    
    #[test]
    fn test_kv_many_operations() -> io::Result<()> {
        let temp_file = NamedTempFile::new()?;
        let mut kv = KV::new(temp_file.path());
        
        kv.open()?;
        
        // Perform many operations to test B-tree behavior
        for i in 0..100 {
            let key = format!("key{:03}", i);
            let value = format!("value{:03}", i);
            kv.set(key.as_bytes(), value.as_bytes())?;
        }
        
        // Verify all keys
        for i in 0..100 {
            let key = format!("key{:03}", i);
            let expected_value = format!("value{:03}", i);
            assert_eq!(kv.get(key.as_bytes()), Some(expected_value.as_bytes().to_vec()));
        }
        
        // Delete some keys first
        for i in 0..100 {
            if i % 3 == 0 {
                let key = format!("key{:03}", i);
                assert_eq!(kv.delete(key.as_bytes())?, true);
            }
        }
        
        // Then update some keys (only if they weren't deleted)
        for i in 0..100 {
            if i % 7 == 0 && i % 3 != 0 {  // Skip keys that were deleted
                let key = format!("key{:03}", i);
                let updated_value = format!("updated{:03}", i);
                kv.set(key.as_bytes(), updated_value.as_bytes())?;
            }
        }
        
        // Verify all operations
        for i in 0..100 {
            let key = format!("key{:03}", i);
            
            if i % 3 == 0 {
                // Deleted keys
                assert_eq!(kv.get(key.as_bytes()), None);
            } else if i % 7 == 0 {
                // Updated keys (not deleted)
                let updated_value = format!("updated{:03}", i);
                assert_eq!(kv.get(key.as_bytes()), Some(updated_value.as_bytes().to_vec()));
            } else {
                // Unchanged keys
                let expected_value = format!("value{:03}", i);
                assert_eq!(kv.get(key.as_bytes()), Some(expected_value.as_bytes().to_vec()));
            }
        }
        
        kv.close();
        Ok(())
    }
    
    #[test]
    fn test_kv_unicode_keys_and_values() -> io::Result<()> {
        let temp_file = NamedTempFile::new()?;
        let mut kv = KV::new(temp_file.path());
        
        kv.open()?;
        
        // Unicode keys and values
        let test_cases = [
            ("中文键", "中文值"),
            ("русский ключ", "русское значение"),
            ("مفتاح عربي", "قيمة عربية"),
            ("ελληνικό κλειδί", "ελληνική αξία"),
            ("日本語キー", "日本語値"),
            ("한국어 키", "한국어 값")
        ];
        
        // Set keys and values
        for (key, value) in &test_cases {
            kv.set(key.as_bytes(), value.as_bytes())?;
        }
        
        // Verify retrieval
        for (key, value) in &test_cases {
            assert_eq!(kv.get(key.as_bytes()), Some(value.as_bytes().to_vec()));
        }
        
        // Delete some keys
        kv.delete(test_cases[1].0.as_bytes())?;
        kv.delete(test_cases[3].0.as_bytes())?;
        
        // Verify after deletion
        assert_eq!(kv.get(test_cases[0].0.as_bytes()), Some(test_cases[0].1.as_bytes().to_vec()));
        assert_eq!(kv.get(test_cases[1].0.as_bytes()), None);
        assert_eq!(kv.get(test_cases[2].0.as_bytes()), Some(test_cases[2].1.as_bytes().to_vec()));
        assert_eq!(kv.get(test_cases[3].0.as_bytes()), None);
        
        kv.close();
        Ok(())
    }
    
    #[test]
    fn test_kv_edge_cases() -> io::Result<()> {
        let temp_file = NamedTempFile::new()?;
        let mut kv = KV::new(temp_file.path());
        
        kv.open()?;
        
        // Empty value
        kv.set(b"empty_value_key", b"")?;
        assert_eq!(kv.get(b"empty_value_key"), Some(vec![]));
        
        // Binary data
        let binary_data = vec![0u8, 1, 2, 3, 255, 254, 253, 252];
        kv.set(b"binary_key", &binary_data)?;
        assert_eq!(kv.get(b"binary_key"), Some(binary_data));
        
        // Maximum allowed key size (should not panic)
        let max_key = vec![b'k'; BTREE_MAX_KEY_SIZE];
        let value = b"max_key_value";
        kv.set(&max_key, value)?;
        assert_eq!(kv.get(&max_key), Some(value.to_vec()));
        
        // Maximum allowed value size (should not panic)
        let key = b"max_value_key";
        let max_value = vec![b'v'; BTREE_MAX_VAL_SIZE];
        kv.set(key, &max_value)?;
        assert_eq!(kv.get(key), Some(max_value));
        
        kv.close();
        Ok(())
    }
}

// DATA PERSISTENCE SYSTEM

struct MmapInfo {
    file_size : u64,
    total_mapped : u64,
    chunks : Vec<MmapMut>
}

struct PageInfo {
    flushed : u64,
    temp : Vec<Vec<u8>>
}

pub struct KV {
    path : String,
    file : Option<File>,
    mmap : MmapInfo,
    page : PageInfo,
    tree : BTree
}

impl KV {
    pub fn new(path : impl AsRef<Path>) -> Self {
        Self {
            path : path.as_ref().to_string_lossy().to_string(),
            file : None,
            mmap : MmapInfo {
                file_size : 0,
                total_mapped : 0,
                chunks : Vec::new()
            },
            page : PageInfo {
                flushed : 0,
                temp : Vec::new()
            },
            tree : BTree::new()
        }
    }

    fn mmap_init(file : &File) -> io::Result<(usize, MmapMut)> {
        let file_size = file.metadata()?.len() as usize;
        if file_size == 0 {
            return Ok((0, unsafe { MmapOptions::new().len(BTREE_PAGE_SIZE).map_mut(file)?}))
        }

        let mmap_size = if file_size < BTREE_PAGE_SIZE {
            BTREE_PAGE_SIZE
        } else {
            file_size
        };

        let mmap = unsafe {
            MmapOptions::new().len(mmap_size).map_mut(file)?
        };

        Ok((file_size, mmap))
    }

    fn extend_mmap(&mut self, npages : usize) -> io::Result<()>{
        let required_size = npages * BTREE_PAGE_SIZE;
        if self.mmap.total_mapped >= required_size as u64 {
            return Ok(());
        }
        let new_size = self.mmap.total_mapped.max(BTREE_PAGE_SIZE as u64);
        let file = self.file.as_ref().unwrap();

        let chunk = unsafe {
            MmapOptions::new()
                .offset(self.mmap.total_mapped as u64)
                .len(new_size as usize)
                .map_mut(file)?
        };

        self.mmap.total_mapped += new_size as u64;
        self.mmap.chunks.push(chunk);
        Ok(())
    }

    fn page_get(&self , ptr : u64) -> BNode {
        let mut start = 0u64;

        for chunk in &self.mmap.chunks {
            let end = start + (chunk.len() / BTREE_PAGE_SIZE) as u64;
            if ptr < end {
                let offset = BTREE_PAGE_SIZE * (ptr - start) as usize;
                let page_data : &[u8] = &chunk[offset..offset + BTREE_PAGE_SIZE];
                // ATTENTION
                return BNode::from_bytes(page_data);
            }
            start = end;          
        }
        panic!("Page not found");
    }
    // Allocate a new page
    fn page_new(&mut self, node: BNode) -> u64 {
        assert!(node.data.len() <= BTREE_PAGE_SIZE);
        let ptr = self.page.flushed + self.page.temp.len() as u64;
        self.page.temp.push(node.data);
        ptr
    }

    // Deallocate a page (placeholder)
    fn page_del(&mut self, _ptr: u64) {
        // TODO: implement free list
    }

    // Load master page
    fn master_load(&mut self) -> io::Result<()> {
        if self.mmap.file_size == 0 {
            // Empty file, master page will be created on first write
            self.page.flushed = 1; // reserved for master page
            return Ok(());
        }

        let data = &self.mmap.chunks[0][0..32];
        let mut cursor = std::io::Cursor::new(data);
        
        // Skip signature for now, read root and used pages
        cursor.set_position(16);
        let root = cursor.read_u64::<LittleEndian>()?;
        let used = cursor.read_u64::<LittleEndian>()?;

        // Verify signature
        if &data[0..16] != DB_SIG {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad signature"));
        }

        // Verify bounds
        let max_pages = (self.mmap.file_size / BTREE_PAGE_SIZE as u64) as u64;
        if !(1 <= used && used <= max_pages) || !(root < used) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad master page"));
        }

        self.tree.root = root;
        self.page.flushed = used;
        Ok(())
    }

    // Store master page atomically
    fn master_store(&mut self) -> io::Result<()> {
        let mut data = [0u8; 32];
        
        // Write signature
        data[0..16].copy_from_slice(DB_SIG);
        
        // Write root and used pages
        let mut cursor = std::io::Cursor::new(&mut data[16..]);
        cursor.write_u64::<LittleEndian>(self.tree.root)?;
        cursor.write_u64::<LittleEndian>(self.page.flushed)?;

        // Use pwrite for atomic update
        let file = self.file.as_mut().unwrap();
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&data)?;
        
        Ok(())
    }

    // Extend file size
    fn extend_file(&mut self, npages: usize) -> io::Result<()> {
        let file_pages = self.mmap.file_size / BTREE_PAGE_SIZE as u64;
        if file_pages >= npages as u64 {
            return Ok(());
        }

        let mut target_pages = file_pages;
        while target_pages < npages as u64 {
            let inc = (target_pages / 8).max(1);
            target_pages += inc;
        }

        let new_size = target_pages * BTREE_PAGE_SIZE as u64;
        let file = self.file.as_mut().unwrap();
        
        // Use fallocate equivalent (set_len)
        file.set_len(new_size as u64)?;
        self.mmap.file_size = new_size;
        
        Ok(())
    }

    // Open database
    pub fn open(&mut self) -> io::Result<()> {
        // Open or create file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&self.path)?;

        // Initialize memory mapping
        let (file_size, chunk) = Self::mmap_init(&file)?;
        self.mmap.file_size = file_size as u64;
        self.mmap.total_mapped = chunk.len() as u64;
        self.mmap.chunks = vec![chunk];

        // Store file handle
        self.file = Some(file);

        // Configure B-tree callbacks
        let kv_ptr = self as *mut KV;
        
        // Set the get function to use page_get
        self.tree.get = Box::new(move |ptr| {
            let kv = unsafe { &*kv_ptr };
            kv.page_get(ptr)
        });
        
        // Set the new function to use page_new
        let kv_ptr = self as *mut KV;
        self.tree.new = Box::new(move |node| {
            let kv = unsafe { &mut *kv_ptr };
            kv.page_new(node)
        });
        
        // Set the del function to use page_del
        let kv_ptr = self as *mut KV;
        self.tree.del = Box::new(move |ptr| {
            let kv = unsafe { &mut *kv_ptr };
            kv.page_del(ptr)
        });

        // Load master page
        self.master_load()?;

        Ok(())
    }

    // Close database
    pub fn close(&mut self) {
        self.mmap.chunks.clear();
        self.file = None;
    }

    // Write pending pages to file
    fn write_pages(&mut self) -> io::Result<()> {
        let npages = self.page.flushed as usize + self.page.temp.len();
        
        // Extend file and mmap if needed
        self.extend_file(npages)?;
        self.extend_mmap(npages)?;

        // Copy data to mapped memory
        for (i, page_data) in self.page.temp.iter().enumerate() {
            let ptr = self.page.flushed + i as u64;
            
            // Find the correct chunk and offset
            let mut start = 0u64;
            for chunk in &mut self.mmap.chunks {
                let chunk_pages = chunk.len() / BTREE_PAGE_SIZE;
                let end = start + chunk_pages as u64;
                
                if ptr >= start && ptr < end {
                    // Found the correct chunk
                    let offset = BTREE_PAGE_SIZE * (ptr - start) as usize;
                    
                    // Copy the page data to the mmap
                    if offset + BTREE_PAGE_SIZE <= chunk.len() {
                        chunk[offset..offset + page_data.len()].copy_from_slice(page_data);
                    }
                    break;
                }
                
                start = end;
            }
        }

        Ok(())
    }

    fn sync_pages(&mut self) -> io::Result<()> {
        // Flush data to disk
        {
            let file = self.file.as_mut().unwrap();
            file.sync_all()?;
        }
    
        // Update counters
        self.page.flushed += self.page.temp.len() as u64;
        self.page.temp.clear();
    
        // Update and flush master page
        self.master_store()?;
    
        // Re-emprunt de file, maintenant autorisé
        {
            let file = self.file.as_mut().unwrap();
            file.sync_all()?;
        }
    
        Ok(())
    }

    // Flush all pending pages
    fn flush_pages(&mut self) -> io::Result<()> {
        self.write_pages()?;
        self.sync_pages()?;
        Ok(())
    }

    // Public API methods
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.tree.get(key)
    }

    pub fn set(&mut self, key: &[u8], val: &[u8]) -> io::Result<()> {
        self.tree.insert(key, val);
        self.flush_pages()
    }

    pub fn delete(&mut self, key: &[u8]) -> io::Result<bool> {
        let deleted = self.tree.delete(key);
        if deleted {
            self.flush_pages()?;
        }
        Ok(deleted)
    }
}


/// Free List : Reusing pages

struct FreeList {
    head : u64,
    get : Box<dyn Fn(u64) -> BNode>,
    new : Box<dyn Fn(BNode) -> u64>,
    used : Box<dyn Fn(u64) -> bool>,
}

impl FreeList {
    pub fn Total(&self) -> i64 {
        return self.head as i64;
        // * TODO REAL IMPL
    }

    pub fn Get(&self , topn : i64) -> u64 {
        return 0;
        // * TODO REAL IMPL
    }

    pub fn Update(&self , popn : i64 , ptr : Vec<u64>) -> u64 {
        // * TODO REAL IMPL
    }



}

