# Data Storage Layer

### Data Storage Layer

The CountYourWords system relies on a straightforward data storage layer to manage and store text files. This layer is crucial for reading, processing, and storing word counts efficiently.

#### File System Usage

CountYourWords primarily uses the file system to store text files that need to be processed. The system reads these files from disk and processes their contents to count words and sort them.

##### Example File Structure

The project includes several test files located in `src/test/textTests/`:

- `emptyFile.txt`: An empty file used for testing edge cases.
- `exampleFile.txt`: A sample text file containing multiple lines of text.
- `nonPeriodFile.txt`: A file without periods, which is handled by the system.
- `validFile.txt`: A valid text file with typical content.

##### Code Example: Reading a File

Below is an example of how the `CountYourWords` class reads a file into an `ArrayList<String>`:

```java
public static ArrayList<String> readFile(String filePath) {
    ArrayList<String> fileLines = new ArrayList<>();
    try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
        String line;
        while ((line = br.readLine()) != null) {
            fileLines.add(line);
        }
    } catch (IOException e) {
        e.printStackTrace();
    }
    return fileLines;
}
```

**Source:** `CountYourWords.java`, PK: 5eaef14f997e9bade8f52072d6f161e7

This method reads each line from the specified file and adds it to an `ArrayList`. It handles exceptions gracefully, ensuring that any I/O errors are logged.

#### Database Usage

CountYourWords does not use a traditional database for storing word counts. Instead, it uses in-memory data structures such as `HashMap` to store and manage word counts efficiently.

##### Example Code: Counting Words

The following code snippet demonstrates how the `CountYourWords` class counts words in a list of strings:

```java
public static Pair<Integer, HashMap<String, Integer>> count(ArrayList<String> fileLines) {
    HashMap<String, Integer> wordCounts = new HashMap<>();
    for (String line : fileLines) {
        String[] words = line.split("\\W+");
        for (String word : words) {
            word = word.toLowerCase();
            if (!word.isEmpty()) {
                wordCounts.put(word, wordCounts.getOrDefault(word, 0) + 1);
            }
        }
    }
    return new Pair<>(wordCounts.size(), wordCounts);
}
```

**Source:** `CountYourWords.java`, PK: 5eaef14f997e9bade8f52072d6f161e7

This method splits each line into words, counts their occurrences, and stores them in a `HashMap`. The `Pair` class is used to return both the total number of unique words and the word count map.

#### Summary

The CountYourWords system's data storage layer is designed to efficiently read text files from disk and process their contents using in-memory data structures. This approach ensures that the system can handle large datasets without significant performance degradation. The use of `HashMap` for storing word counts allows for quick lookups and updates, making it an ideal choice for this application.