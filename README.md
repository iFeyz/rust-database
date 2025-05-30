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

## Français

### Aperçu

Ce projet implémente une base de données clé-valeur persistante en Rust utilisant des structures de données B-tree pour un stockage et une récupération efficaces des données. La base de données prend en charge les opérations de base comme la définition de valeurs, la récupération de valeurs et la suppression de paires clé-valeur. Elle assure la durabilité en persistant les données sur le disque et implémente un mécanisme de liste libre pour une réutilisation efficace des pages.

### Composants Principaux

#### Implémentation B-tree
- **BNode**: Représente un nœud dans la structure B-tree, contenant des clés, des valeurs et des pointeurs
- **BTree**: Gère la structure B-tree avec des opérations d'insertion, de récupération et de suppression
- Gestion personnalisée des nœuds avec des fonctions pour diviser et fusionner les nœuds si nécessaire

#### Magasin Clé-Valeur (KV)
- Interface principale pour les opérations de la base de données
- Fournit des méthodes API simples: `set()`, `get()`, et `delete()`
- Utilise des fichiers mappés en mémoire pour des E/S efficaces
- Persiste les changements sur le disque, assurant la durabilité des données

#### Liste Libre
- Gère les pages réutilisables dans le fichier de base de données
- Implémente un système d'allocation mémoire simple pour les pages disque
- Optimise le stockage en réutilisant les pages libérées avant d'en allouer de nouvelles

### Format de Fichier

Le fichier de base de données a la structure suivante:
- Page d'en-tête avec signature, pointeur racine, nombre de pages utilisées et tête de liste libre
- Nœuds B-tree stockés comme pages de taille fixe (4Ko)
- Nœuds de liste libre pour suivre les pages supprimées

### Exemple d'Utilisation

```rust
// Créer une instance de base de données
let path = Path::new("ma_base_de_donnees.db");
let mut database = KV::new(path);

// Ouvrir la base de données
database.open()?;

// Stocker des données
database.set(b"utilisateur:1001", b"Jean Dupont")?;
database.set(b"utilisateur:1002", b"Marie Martin")?;

// Récupérer des données
if let Some(value) = database.get(b"utilisateur:1001") {
    println!("Utilisateur 1001: {}", String::from_utf8_lossy(&value));
}

// Supprimer des données
database.delete(b"utilisateur:1001")?;

// Fermer la base de données
database.close();
```

### Détails Techniques

1. **Structure B-tree**:
   - Les nœuds internes contiennent des pointeurs vers les nœuds enfants
   - Les nœuds feuilles stockent les paires clé-valeur
   - La structure auto-équilibrante assure des opérations en O(log n)
   - Les nœuds se divisent lorsqu'ils deviennent trop grands

2. **Gestion de la Mémoire**:
   - Utilise des fichiers mappés en mémoire pour des E/S efficaces
   - Maintient une liste libre pour réutiliser les pages supprimées
   - Fait croître le fichier selon les besoins, par incréments pour réduire la fragmentation

3. **Concurrence**:
   - L'implémentation actuelle est mono-thread
   - Les améliorations futures pourraient ajouter la sécurité des threads

4. **Durabilité**:
   - Les mises à jour sont écrites sur le disque avant que les opérations ne se terminent
   - La page maître assure la cohérence après les crashs
   - Toutes les opérations assurent des mises à jour atomiques du fichier de base de données

### Considérations de Performance

- Taille de page fixe de 4Ko, optimisée pour la plupart des systèmes de fichiers
- La structure B-tree assure une complexité temporelle logarithmique pour les opérations
- Le mappage mémoire fournit un accès efficace aux pages de la base de données
- La liste libre réduit le gaspillage d'espace disque et la fragmentation


