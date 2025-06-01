# Key-Value Database Implementation in Rust

## English

### Overview

This project implements a persistent key-value store in Rust using B-tree data structures for efficient data storage and retrieval. The database supports basic operations like setting values, retrieving values, and deleting key-value pairs. It provides durability by persisting data to disk and implements a free list mechanism for efficient page reuse.

### Core Components

#### B-tree Implementation
- **BNode**: Represents a node in the B-tree structure, containing keys, values, and pointers
- **BTree**: Manages the B-tree structure with operations for insertion, retrieval, and deletion
- Custom node management with functions for splitting and merging nodes when necessary

#### Key-Value Store (KV)
- Main interface for the database operations
- Provides simple API methods: `set()`, `get()`, and `delete()`
- Uses memory-mapped files for efficient I/O
- Persists changes to disk, ensuring data durability

#### Free List
- Manages reusable pages in the database file
- Implements a simple memory allocation system for disk pages
- Optimizes storage by reusing freed pages before allocating new ones

### File Format

The database file has the following structure:
- Header page with signature, root pointer, used pages count, and free list head
- B-tree nodes stored as fixed-size pages (4KB)
- Free list nodes for tracking deleted pages

### Usage Example

```rust
// Create a database instance
let path = Path::new("my_database.db");
let mut database = KV::new(path);

// Open the database
database.open()?;

// Store data
database.set(b"user:1001", b"John Doe")?;
database.set(b"user:1002", b"Jane Smith")?;

// Retrieve data
if let Some(value) = database.get(b"user:1001") {
    println!("User 1001: {}", String::from_utf8_lossy(&value));
}

// Delete data
database.delete(b"user:1001")?;

// Close the database
database.close();
```

### Technical Details

1. **B-tree Structure**:
   - Internal nodes contain pointers to child nodes
   - Leaf nodes store key-value pairs
   - Auto-balancing structure ensures O(log n) operations
   - Nodes split when they become too large

2. **Memory Management**:
   - Uses memory-mapped files for efficient I/O
   - Maintains a free list for reusing deleted pages
   - Grows the file as needed, in increments to reduce fragmentation

3. **Concurrency**:
   - Current implementation is single-threaded
   - Future improvements could add thread-safety

4. **Durability**:
   - Updates are written to disk before operations complete
   - Master page ensures consistency after crashes
   - All operations ensure atomic updates to the database file

### Performance Considerations

- Fixed page size of 4KB, optimized for most filesystems
- B-tree structure ensures logarithmic time complexity for operations
- Memory mapping provides efficient access to database pages
- Free list reduces disk space waste and fragmentation

---


# DevLog: Recent Feature Additions

## Range Queries (May 2023)

Range queries enable the retrieval of multiple key-value pairs that fall within a specified range, allowing for efficient data exploration and analysis.

### Key Components:
- **BTreeIter**: A new cursor-like structure that traverses the B-tree in a controlled manner
- **Range Comparison Operators**: 
  - `CMP_GE` (>=): Greater than or equal to
  - `CMP_GT` (>): Greater than
  - `CMP_LT` (<): Less than
  - `CMP_LE` (<=): Less than or equal to

### Usage Example:
```rust
// Range query: retrieve all keys between "key03" and "key07" (inclusive)
let mut iter = btree.seek(b"key03", CMP_GE);
while iter.Valid() {
    let (key, value) = iter.Deref();
    // Stop if we've passed the upper bound
    if !cmpOK(&key, CMP_LE, b"key07") {
        break;
    }
    println!("{} = {}", 
        String::from_utf8_lossy(&key), 
        String::from_utf8_lossy(&value));
    iter.Next();
}
```

### Benefits:
- Efficient traversal of sorted data without loading all records into memory
- Ability to perform range-based analytics directly on the database
- Support for pagination by using iterators to fetch fixed-size batches of results

## Order-Preserving Encoding (May 2023)

A critical feature enabling range queries is our order-preserving encoding system, which ensures that lexicographic comparison of encoded values matches the logical ordering of the original values.

### Encoding Features:
- **Integer Encoding**: Transforms integers to maintain sort order (including negative numbers)
- **String Encoding**: Escapes null bytes and control characters to preserve lexicographic ordering
- **Composite Key Encoding**: Maintains correct ordering for multi-part keys

### Key Benefits:
- **Correct Sorting**: Ensures `int64` values like -100, -1, 0, 1, 100 are properly ordered after encoding
- **Binary-Safe**: Properly handles binary data containing null bytes or other special characters
- **Efficient Comparison**: Allows direct byte comparison without decoding, improving performance

## Scanner Implementation (May 2023)

The Scanner provides a high-level interface for database queries, especially for the tabular database layer built on top of the key-value store.

### Features:
- **Composite Key Support**: Handles queries against multi-column primary keys
- **Customizable Range Boundaries**: Supports various comparison operations for range limits
- **Integration with TableDef**: Works with the schema information from table definitions

### Technical Details:
- Translates high-level query constraints into low-level B-tree operations
- Manages the lifecycle of iterators and properly handles their creation and cleanup
- Provides a uniform interface regardless of the underlying index structure

## Secondary Indexes (WIP)

*Note: This feature is currently under development*

Secondary indexes will enhance query capabilities by allowing efficient lookups on non-primary key columns.

### Planned Features:
- **Automatic Index Maintenance**: Indexes automatically updated when records change
- **Multi-Column Indexes**: Support for composite indexes on multiple columns
- **Index-Based Query Optimization**: Query planner that selects the most efficient index

### Current Status:
- Base infrastructure for secondary indexes has been implemented
- Testing with various data patterns and query types is ongoing
- Performance optimization and space efficiency improvements in progress

---

These new features significantly enhance the capabilities of our database, bringing it closer to a full-featured database management system while maintaining the simplicity and efficiency of the original key-value store design.
