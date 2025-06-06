# Testing Strategy

## Testing Strategy

The CountYourWords project employs a comprehensive testing strategy to ensure the reliability and correctness of its components. This strategy includes both unit tests and integration tests, leveraging JUnit 4.13.2 for assertions.

### Unit Tests

Unit tests are designed to validate individual methods or functions within the application. The primary focus is on the `CountYourWords` class and its associated helper methods. Below are some key unit test cases:

#### Sorting Algorithm
The sorting algorithm in `CountYourWords.sort()` method is tested using JUnit. Here’s an example of a unit test for sorting an empty map:

```java
@Test
public void sortEmptyTest() {
    HashMap<String, Integer> emptyMap = new HashMap<>();
    ArrayList<String> sortedArray = CountYourWords.sort(emptyMap);
    assertTrue("Array should be empty", sortedArray.isEmpty());
}
```

This test ensures that the sorting method returns an empty list when provided with an empty map.

#### Word Counting
The `CountYourWords.count()` method is tested to ensure it correctly counts words in different scenarios. Here’s a unit test for counting words in a single line:

```java
@Test
public void testSingleLine() {
    ArrayList<String> fileLines = new ArrayList<>();
    fileLines.add("Hello world");

    Pair result = CountYourWords.count(fileLines);

    assertEquals("Total words should be 2", 2, result.getFirst());

    HashMap<String, Integer> expectedCounts = new HashMap<>();
    expectedCounts.put("hello", 1);
    expectedCounts.put("world", 1);

    assertEquals("Word counts should match expected counts", expectedCounts, result.getSecond());
}
```

This test verifies that the word count is accurate and that the map of word counts contains the correct entries.

### Integration Tests

Integration tests are used to verify the interaction between different components or modules. For CountYourWords, integration tests focus on testing the end-to-end functionality using real text files.

#### Testing with Empty File
The `CountYourWords.count()` method is tested with an empty file:

```java
@Test
public void testEmptyFile() {
    ArrayList<String> fileLines = new ArrayList<>();
    Pair result = CountYourWords.count(fileLines);

    assertEquals("Total words should be 0", 0, result.getFirst());
    assertTrue("Word counts map should be empty", result.getSecond().isEmpty());
}
```

This test ensures that the method handles an empty file correctly.

#### Testing with Multiple Lines
The `CountYourWords.count()` method is tested with multiple lines of text:

```java
@Test
public void testMultipleLines() {
    ArrayList<String> fileLines = new ArrayList<>();
    fileLines.add("Hello world");
    fileLines.add("This is a test.");
    fileLines.add("World of Java!");

    Pair result = CountYourWords.count(fileLines);

    assertEquals("Total words should be 9", 9, result.getFirst());

    HashMap<String, Integer> expectedCounts = new HashMap<>();
    expectedCounts.put("hello", 1);
    expectedCounts.put("world", 2);
    expectedCounts.put("this", 1);
    expectedCounts.put("is", 1);
    expectedCounts.put("a", 1);
    expectedCounts.put("test", 1);
    expectedCounts.put("of", 1);
    expectedCounts.put("java", 1);

    assertEquals("Word counts should match expected counts", expectedCounts, result.getSecond());
}
```

This test ensures that the method correctly counts words across multiple lines.

### Additional Notes

- **Dependencies:** The testing strategy relies on JUnit for assertions and Hamcrest for more expressive matchers.
- **Test Data:** Test files are located in `CountYourWords/src/test/textTests/`, including `emptyFile.txt`, `exampleFile.txt`, `nonPeriodFile.txt`, and `validFile.txt`.

This comprehensive testing approach ensures that the CountYourWords project is robust and reliable, providing accurate word counts and sorted results.