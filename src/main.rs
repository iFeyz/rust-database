use std::fs::{File, rename};
use std::io::{Write, Result};
use std::path::{Path, PathBuf};
use std::cmp::Ordering;

const HEADER: usize = 4;
const BTREE_PAGE_SIZE: usize = 4096;
const BTREE_MAX_KEY_SIZE: usize = 1000;
const BTREE_MAX_VAL_SIZE: usize = 3000;
const BNODE_NODE: u16 = 1; // internal nodes without values
const BNODE_LEAF: u16 = 2; // leaf nodes with values

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
        if index == self.nkeys() {
            // Special case for finding the end position
            let offset_section_size = 2 * self.nkeys() as usize;
            return (HEADER + 8 * self.nkeys() as usize + offset_section_size) as u16;
        }
        
        assert!(index < self.nkeys());
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
        let offset = dstBegin + old.getOffset(srcOld + i - 1)  - srcBegin;
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
    match node.btype() {
        BNODE_LEAF => {
            // Leaf node, node.getKey(index) <= key
            if key == node.getKey(index) {
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
    nodeAppendRange(new, old, index, index + 1, old.nkeys() - index -1);
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
    let newnkeys = old.nkeys() + inc -1;
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
        if self.root == 0 {
            // Create root node
            let mut root = BNode::new(BTREE_PAGE_SIZE);
            root.setHeader(BNODE_LEAF, 1);
            nodeAppendKv(&mut root, 0, 0, key, val);
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
        println!("GET: Starting at root node type: {}, nkeys: {}", node.btype(), node.nkeys());
        
        loop {
            let idx = nodeLookupLe(&node, key);
            println!("GET: nodeLookupLe for key {:?} returned idx: {}", key, idx);
            
            match node.btype() {
                BNODE_LEAF => {
                    // For leaf nodes, we need to check if we found the exact key
                    if idx < node.nkeys() && node.getKey(idx) == key {
                        println!("GET: Found key at idx {} in leaf node", idx);
                        return Some(node.getVal(idx).to_vec());
                    } else {
                        println!("GET: Key not found in leaf node");
                        return None;
                    }
                }
                BNODE_NODE => {
                    // For internal nodes, we follow the pointer
                    println!("GET: Following pointer at idx {} in internal node", idx);
                    let ptr = node.getPtr(idx);
                    node = (self.get)(ptr);
                    println!("GET: New node type: {}, nkeys: {}", node.btype(), node.nkeys());
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
        
        // Debug the root state after first insert
        let root_node = (btree.get)(btree.root);
        println!("After key1: Root type: {}, nkeys: {}", root_node.btype(), root_node.nkeys());
        
        if root_node.nkeys() > 0 {
            for i in 0..root_node.nkeys() {
                println!("  Key at {}: {:?}", i, root_node.getKey(i));
            }
        }
        
        // Test first retrieval
        let result1 = btree.get(b"key1");
        println!("get(key1) returned: {:?}", result1);
        assert_eq!(result1, Some(b"value1".to_vec()));
        
        // Insert second key-value pair
        println!("\nInserting key2=value2");
        btree.insert(b"key2", b"value2");
        
        // Debug the root state after second insert
        let root_node = (btree.get)(btree.root);
        println!("After key2: Root type: {}, nkeys: {}", root_node.btype(), root_node.nkeys());
        
        if root_node.nkeys() > 0 {
            for i in 0..root_node.nkeys() {
                println!("  Key at {}: {:?}", i, root_node.getKey(i));
            }
            
            // Check if the keys are found with our lookup function
            let idx1 = nodeLookupLe(&root_node, b"key1");
            let idx2 = nodeLookupLe(&root_node, b"key2");
            println!("nodeLookupLe for key1: {}, key2: {}", idx1, idx2);
        }
        
        // Test second retrieval
        let result2 = btree.get(b"key2");
        println!("get(key2) returned: {:?}", result2);
        assert_eq!(result2, Some(b"value2".to_vec()));
        
        // Insert third key-value pair if the second one works
        if result2.is_some() {
            println!("\nInserting key3=value3");
            btree.insert(b"key3", b"value3");
            let result3 = btree.get(b"key3");
            println!("get(key3) returned: {:?}", result3);
            assert_eq!(result3, Some(b"value3".to_vec()));
        }
        
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
}







//* Atomic write Data to file
pub fn atomic_write<P : AsRef<Path>>(path : P, data : &[u8]) -> Result<()> {
    let target_path = path.as_ref();
    let temp_path = create_temp_file(target_path)?;
    {
        let mut temp_file = File::create(&temp_path)?;
        temp_file.write_all(data)?;
        temp_file.sync_all()?;
    }
    rename(temp_path, target_path)?;
    Ok(())
}

pub fn atomic_write_str<P : AsRef<Path>>(path : P, content : &str) -> Result<()> {
    atomic_write(path, content.as_bytes())
}

pub fn create_temp_file(target: &Path) -> Result<PathBuf> {
    let parent = target.parent().unwrap_or(Path::new("."));
    let filename = target.file_name().and_then(|name| name.to_str()).unwrap_or("temp");
    let temp_name = format!(".{}.tmp.{}", filename, std::process::id());
    Ok(parent.join(temp_name))
}

fn main() {
    let path = "test.txt";
    let data = b"Hello, world!";
    if let Err(e) = atomic_write(path, data) {
        eprintln!("Error writing file: {}", e);
    }
    let content = "Hello, world!";
    if let Err(e) = atomic_write_str(path, content) {
        eprintln!("Error writing file: {}", e);
    }
}

