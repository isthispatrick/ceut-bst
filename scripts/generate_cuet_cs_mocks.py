from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pandas as pd

from init_cuet_cs import PROCESSED_DIR


QUESTION_BANK = [
    ("Section A Common Core", "Database Concepts", "candidate primary alternate foreign keys", "Which key uniquely identifies each record in a relation and cannot contain NULL values?", ["Foreign key", "Primary key", "Alternate key", "Composite attribute"], "B", "A primary key uniquely identifies each tuple and cannot be NULL."),
    ("Section A Common Core", "Database Concepts", "candidate primary alternate foreign keys", "A key in one table that refers to the primary key of another table is called a:", ["Candidate key", "Foreign key", "Alternate key", "Super domain"], "B", "A foreign key creates a reference between two relations."),
    ("Section A Common Core", "Database Concepts", "relational model", "In the relational model, one row of a table is called a:", ["Domain", "Attribute", "Tuple", "Schema"], "C", "A tuple is a row in a relation."),
    ("Section A Common Core", "Database Concepts", "domain tuple relation", "The set of permitted values for an attribute is called its:", ["Domain", "Tuple", "Relation", "Cardinality"], "A", "Domain means the valid value range for an attribute."),
    ("Section A Common Core", "Database Concepts", "relational algebra selection projection union set difference cartesian product", "Which relational algebra operation selects rows satisfying a condition?", ["Projection", "Selection", "Union", "Cartesian product"], "B", "Selection filters rows; projection selects columns."),
    ("Section A Common Core", "Database Concepts", "relational algebra selection projection union set difference cartesian product", "Which relational algebra operation selects specific columns from a relation?", ["Projection", "Selection", "Set difference", "Union"], "A", "Projection returns chosen attributes/columns."),
    ("Section A Common Core", "Structured Query Language I", "SELECT FROM WHERE", "Which SQL clause specifies the table from which records are retrieved?", ["WHERE", "FROM", "ORDER BY", "HAVING"], "B", "FROM names the table or tables used by the query."),
    ("Section A Common Core", "Structured Query Language I", "DDL DQL DML", "Which of the following is a DDL command?", ["INSERT", "UPDATE", "CREATE", "SELECT"], "C", "CREATE changes schema, so it is DDL."),
    ("Section A Common Core", "Structured Query Language I", "INSERT UPDATE DELETE", "Which SQL command is used to modify existing records?", ["ALTER", "UPDATE", "DROP", "CREATE"], "B", "UPDATE changes values in existing rows."),
    ("Section A Common Core", "Structured Query Language I", "text functions", "Which MySQL function converts a string to uppercase?", ["LCASE()", "UPPER()", "ROUND()", "CURDATE()"], "B", "UPPER() returns uppercase text."),
    ("Section A Common Core", "Structured Query Language II", "aggregate functions", "Which SQL function returns the number of rows?", ["SUM()", "AVG()", "COUNT()", "MAX()"], "C", "COUNT() is used for row counts."),
    ("Section A Common Core", "Structured Query Language II", "COUNT star", "COUNT(*) in SQL counts:", ["Only non-NULL values in one column", "All selected rows", "Only unique rows", "Only text values"], "B", "COUNT(*) counts rows, including rows with NULL column values."),
    ("Section A Common Core", "Structured Query Language II", "GROUP BY HAVING ORDER BY", "Which clause filters groups after GROUP BY?", ["WHERE", "HAVING", "ORDER BY", "FROM"], "B", "HAVING filters grouped results."),
    ("Section A Common Core", "Structured Query Language II", "GROUP BY HAVING ORDER BY", "Correct SQL clause order is:", ["SELECT, WHERE, FROM, GROUP BY", "SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY", "FROM, SELECT, ORDER BY, WHERE", "WHERE, SELECT, FROM, HAVING"], "B", "This is the usual logical writing order in school-level SQL."),
    ("Section A Common Core", "Structured Query Language II", "union intersection minus cartesian product join", "If table A has 3 rows and table B has 4 rows, A Cartesian product B has:", ["7 rows", "12 rows", "1 row", "4 rows"], "B", "Cartesian product row count is m x n."),
    ("Section A Common Core", "Computer Networks", "LAN WAN MAN", "A network limited to a building or campus is generally a:", ["WAN", "LAN", "MAN", "PAN only"], "B", "LAN covers a local area."),
    ("Section A Common Core", "Computer Networks", "network devices", "Which device forwards packets between different networks?", ["Switch", "Router", "Repeater", "Hub"], "B", "A router connects different networks."),
    ("Section A Common Core", "Computer Networks", "topologies", "In which topology is every node connected to a central device?", ["Bus", "Ring", "Star", "Mesh"], "C", "Star topology uses a central hub/switch."),
    ("Section A Common Core", "Computer Networks", "MAC and IP address", "A MAC address is mainly associated with:", ["Application software", "Network interface hardware", "SQL table", "File extension"], "B", "MAC is a hardware/network interface address."),
    ("Section A Common Core", "Computer Networks", "internet vs web", "The World Wide Web is:", ["A service using the internet", "A type of database key", "A local network cable", "A sorting algorithm"], "A", "The web is one service that runs on the internet."),
    ("Section B1 Computer Science", "Exception and File Handling in Python", "try except else finally", "Which block executes whether or not an exception occurs?", ["try", "except", "else", "finally"], "D", "finally is used for cleanup and always executes if reached."),
    ("Section B1 Computer Science", "Exception and File Handling in Python", "raise exceptions", "Which keyword is used to explicitly throw an exception in Python?", ["throw", "raise", "except", "error"], "B", "Python uses raise."),
    ("Section B1 Computer Science", "Exception and File Handling in Python", "pickle", "The pickle module is mainly used for:", ["Sorting lists", "Serializing Python objects", "Creating SQL tables", "Drawing graphs"], "B", "Pickle serializes/deserializes Python objects."),
    ("Section B1 Computer Science", "Exception and File Handling in Python", "file access modes", "Which mode opens a file for binary reading?", ["r", "w", "rb", "ab+"], "C", "rb means read binary."),
    ("Section B1 Computer Science", "Stack", "LIFO", "A stack follows which principle?", ["FIFO", "LIFO", "Random access", "Round robin"], "B", "Stack is Last In First Out."),
    ("Section B1 Computer Science", "Stack", "push pop", "The operation used to insert an element into a stack is:", ["enqueue", "dequeue", "push", "peek only"], "C", "push inserts into a stack."),
    ("Section B1 Computer Science", "Stack", "expression evaluation", "In postfix expression `2 3 +`, the result is:", ["23", "5", "1", "6"], "B", "2 3 + means 2 + 3."),
    ("Section B1 Computer Science", "Stack", "infix to postfix", "The postfix form of `A+B` is:", ["AB+", "+AB", "A+B", "BA+"], "A", "Operator comes after operands in postfix."),
    ("Section B1 Computer Science", "Queue", "FIFO", "A queue follows which principle?", ["LIFO", "FIFO", "Divide and conquer", "Hashing only"], "B", "Queue is First In First Out."),
    ("Section B1 Computer Science", "Queue", "insert delete", "Insertion in a simple queue is generally performed at:", ["Front", "Rear", "Middle", "Top"], "B", "Queue insertion is at rear and deletion is at front."),
    ("Section B1 Computer Science", "Searching", "binary search", "Binary search requires the data to be:", ["Unsorted", "Sorted", "Encrypted", "Only strings"], "B", "Binary search works on sorted data."),
    ("Section B1 Computer Science", "Searching", "sequential search", "Worst-case time complexity of linear search is:", ["O(1)", "O(log n)", "O(n)", "O(n log n)"], "C", "Linear search may check every item."),
    ("Section B1 Computer Science", "Sorting", "bubble sort", "Bubble sort repeatedly compares:", ["Adjacent elements", "Only first and last elements", "Only keys", "Only database rows"], "A", "Bubble sort compares adjacent elements and swaps when needed."),
    ("Section B1 Computer Science", "Sorting", "selection sort", "Selection sort repeatedly selects the:", ["Middle element", "Minimum/maximum element for position", "Random element", "Duplicate key only"], "B", "Selection sort selects the next min/max and places it."),
    ("Section B1 Computer Science", "Sorting", "insertion sort", "Insertion sort builds:", ["A sorted part one element at a time", "A network topology", "A SQL relation only", "A hash collision only"], "A", "Insertion sort inserts each item into the sorted left part."),
    ("Section B1 Computer Science", "Sorting", "hashing", "When two keys map to the same hash location, it is called:", ["Projection", "Collision", "Iteration", "Serialization"], "B", "Hash collision means two keys share a hash slot."),
    ("Section B1 Computer Science", "Understanding Data", "mean median", "The median is:", ["The most frequent value", "The middle value after sorting", "The sum of all values", "The range of values"], "B", "Median is the middle value in sorted data."),
    ("Section B1 Computer Science", "Understanding Data", "standard deviation variance", "Variance measures:", ["Spread of data", "Primary key count", "Network speed only", "Queue length only"], "A", "Variance measures dispersion/spread."),
    ("Section B1 Computer Science", "Data Communication and Security", "protocols", "HTTPS is a secure version of:", ["FTP", "HTTP", "SMTP", "LAN"], "B", "HTTPS secures HTTP communication."),
    ("Section B1 Computer Science", "Data Communication and Security", "firewall http https", "A firewall is used to:", ["Sort records", "Filter network traffic", "Create primary keys", "Convert infix to postfix"], "B", "Firewall filters allowed/blocked traffic."),
    ("Section B1 Computer Science", "Data Communication and Security", "DoS intrusion snooping eavesdropping", "A DoS attack mainly tries to:", ["Improve bandwidth", "Make a service unavailable", "Create a relation", "Serialize an object"], "B", "Denial of Service attacks target availability."),
    ("Section B1 Computer Science", "Data Communication and Security", "bandwidth and data transfer rate", "Bandwidth is generally related to:", ["Data carrying capacity", "File mode only", "Stack top", "Median"], "A", "Bandwidth describes capacity/rate of communication."),
]

QUESTION_BANK += [
    ("Section A Common Core", "Database Concepts", "database vs file system", "Which is a key advantage of a database system over a traditional file system?", ["More data redundancy", "Better data sharing and reduced redundancy", "No need for keys", "Only sequential access"], "B", "DBMS reduces redundancy and supports controlled sharing."),
    ("Section A Common Core", "Database Concepts", "database vs file system", "In a file system, the same customer address stored in many files mainly causes:", ["Data redundancy", "Primary key creation", "Projection", "Packet switching"], "A", "Repeated storage of the same fact is redundancy."),
    ("Section A Common Core", "Database Concepts", "relational model", "The structure of a relation, including its attributes, is called:", ["Tuple", "Schema", "Domain", "Cardinality"], "B", "A schema describes the relation structure."),
    ("Section A Common Core", "Database Concepts", "relational model", "The number of tuples in a relation is called:", ["Degree", "Cardinality", "Domain", "Projection"], "B", "Cardinality is the number of rows/tuples."),
    ("Section A Common Core", "Database Concepts", "domain tuple relation", "The number of attributes in a relation is called:", ["Degree", "Cardinality", "Foreign key", "Selection"], "A", "Degree is the number of columns/attributes."),
    ("Section A Common Core", "Database Concepts", "candidate primary alternate foreign keys", "A candidate key not chosen as the primary key is called:", ["Foreign key", "Alternate key", "Domain", "Tuple"], "B", "Alternate keys are candidate keys not selected as primary."),
    ("Section A Common Core", "Database Concepts", "candidate primary alternate foreign keys", "A primary key should be:", ["Unique and not NULL", "Always text", "Repeated in every row", "A network address"], "A", "Primary keys uniquely identify rows and cannot be NULL."),
    ("Section A Common Core", "Database Concepts", "relational algebra selection projection union set difference cartesian product", "Union of two relations requires them to be:", ["Sorted", "Union compatible", "Encrypted", "In different databases only"], "B", "Union needs compatible attributes/domains."),
    ("Section A Common Core", "Database Concepts", "relational algebra selection projection union set difference cartesian product", "Which operation returns rows present in one relation but not another?", ["Selection", "Projection", "Set difference", "Cartesian product"], "C", "Set difference returns tuples in the first relation but not the second."),
    ("Section A Common Core", "Structured Query Language I", "SQL advantages", "SQL is mainly used to:", ["Communicate between routers", "Store, define, and retrieve relational data", "Convert postfix expressions", "Create viruses"], "B", "SQL works with relational database definition and manipulation."),
    ("Section A Common Core", "Structured Query Language I", "SQL advantages", "A major advantage of SQL is that it is:", ["A low-level machine language", "Declarative and set-oriented", "Only for image files", "Only for stacks"], "B", "SQL specifies what data is needed rather than every procedural step."),
    ("Section A Common Core", "Structured Query Language I", "DDL DQL DML", "Which command belongs to DML?", ["CREATE", "DROP", "INSERT", "SELECT only"], "C", "INSERT is a data manipulation command."),
    ("Section A Common Core", "Structured Query Language I", "DDL DQL DML", "SELECT is commonly classified as:", ["DDL", "DQL", "DCL only", "TCL only"], "B", "SELECT is a data query language command."),
    ("Section A Common Core", "Structured Query Language I", "MySQL database creation", "Which command creates a new database in MySQL?", ["MAKE DATABASE school;", "CREATE DATABASE school;", "NEW DATABASE school;", "INSERT DATABASE school;"], "B", "CREATE DATABASE creates a database."),
    ("Section A Common Core", "Structured Query Language I", "MySQL database creation", "Before creating tables inside a database, which command selects it?", ["USE database_name;", "OPEN database_name;", "SELECT database_name;", "START database_name;"], "A", "USE selects the active database."),
    ("Section A Common Core", "Structured Query Language I", "data types", "Which MySQL data type is suitable for whole numbers?", ["VARCHAR", "DATE", "INT", "CHAR only"], "C", "INT stores integer values."),
    ("Section A Common Core", "Structured Query Language I", "data types", "Which data type is most suitable for variable-length names?", ["VARCHAR", "INT", "DATE", "FLOAT only"], "A", "VARCHAR stores variable-length character data."),
    ("Section A Common Core", "Structured Query Language I", "CREATE DROP ALTER", "Which command changes the structure of an existing table?", ["ALTER TABLE", "UPDATE TABLE", "CHANGE ROW", "ORDER BY"], "A", "ALTER TABLE modifies table structure."),
    ("Section A Common Core", "Structured Query Language I", "CREATE DROP ALTER", "Which command removes a table definition and its data?", ["DELETE TABLE", "DROP TABLE", "CLEAR TABLE", "REMOVE ROW"], "B", "DROP TABLE removes the table object."),
    ("Section A Common Core", "Structured Query Language I", "SELECT FROM WHERE", "Which clause filters rows before grouping?", ["WHERE", "HAVING", "ORDER BY", "CREATE"], "A", "WHERE filters individual rows before GROUP BY."),
    ("Section A Common Core", "Structured Query Language I", "SELECT FROM WHERE", "SELECT name FROM Student WHERE marks > 80 returns:", ["All columns", "Names of students scoring above 80", "Only table structure", "A new database"], "B", "The SELECT list has name and WHERE filters marks."),
    ("Section A Common Core", "Structured Query Language I", "INSERT UPDATE DELETE", "Which command removes selected rows but keeps the table?", ["DROP", "DELETE", "ALTER", "CREATE"], "B", "DELETE removes rows from an existing table."),
    ("Section A Common Core", "Structured Query Language I", "INSERT UPDATE DELETE", "Which command adds a new row to a table?", ["INSERT INTO", "ADD DATABASE", "ALTER COLUMN", "COUNT"], "A", "INSERT INTO adds records."),
    ("Section A Common Core", "Structured Query Language I", "math functions", "Which function returns the rounded value of a number?", ["ROUND()", "UPPER()", "LENGTH()", "CURDATE()"], "A", "ROUND() rounds numeric values."),
    ("Section A Common Core", "Structured Query Language I", "math functions", "ABS(-7) returns:", ["-7", "0", "7", "Error always"], "C", "ABS returns absolute value."),
    ("Section A Common Core", "Structured Query Language I", "text functions", "Which function returns the number of characters in a string?", ["LENGTH()", "ROUND()", "NOW()", "AVG()"], "A", "LENGTH returns string length in many school-level MySQL contexts."),
    ("Section A Common Core", "Structured Query Language I", "text functions", "LOWER('CS') returns:", ["CS", "cs", "Cs", "Error"], "B", "LOWER converts text to lowercase."),
    ("Section A Common Core", "Structured Query Language II", "date functions", "Which function returns the current date?", ["CURDATE()", "COUNT()", "UCASE()", "ROUND()"], "A", "CURDATE returns the current date."),
    ("Section A Common Core", "Structured Query Language II", "date functions", "YEAR('2026-05-22') returns:", ["2026", "05", "22", "0522"], "A", "YEAR extracts the year part."),
    ("Section A Common Core", "Structured Query Language II", "aggregate functions", "Which aggregate function returns the highest value?", ["MIN()", "MAX()", "AVG()", "COUNT()"], "B", "MAX returns the largest value."),
    ("Section A Common Core", "Structured Query Language II", "aggregate functions", "AVG(marks) returns:", ["Total marks", "Average marks", "Number of rows", "Lowest marks"], "B", "AVG returns arithmetic mean."),
    ("Section A Common Core", "Structured Query Language II", "COUNT star", "COUNT(column_name) does not count:", ["Non-NULL values", "NULL values in that column", "Rows with numbers", "Rows with text"], "B", "COUNT(column) ignores NULLs."),
    ("Section A Common Core", "Structured Query Language II", "GROUP BY HAVING ORDER BY", "ORDER BY marks DESC sorts marks in:", ["Ascending order", "Descending order", "Random order", "Grouped order only"], "B", "DESC means descending."),
    ("Section A Common Core", "Structured Query Language II", "GROUP BY HAVING ORDER BY", "Which clause groups rows with the same value?", ["GROUP BY", "ORDER BY", "WHERE only", "DROP"], "A", "GROUP BY forms groups for aggregate calculations."),
    ("Section A Common Core", "Structured Query Language II", "union intersection minus cartesian product join", "An equi-join usually combines tables using:", ["A comparison with equality between related columns", "Only unrelated columns", "A stack", "A firewall"], "A", "Equi-join uses equality condition between matching columns."),
    ("Section A Common Core", "Structured Query Language II", "union intersection minus cartesian product join", "Which result contains rows common to both compatible relations?", ["UNION", "INTERSECTION", "MINUS", "CARTESIAN PRODUCT"], "B", "Intersection gives common rows."),
    ("Section A Common Core", "Computer Networks", "network evolution", "ARPANET is historically important because it was:", ["An early packet-switched network", "A SQL command", "A sorting method", "A file mode"], "A", "ARPANET is a predecessor of the internet."),
    ("Section A Common Core", "Computer Networks", "network evolution", "The internet is best described as:", ["A single LAN cable", "A network of networks", "A database table", "A stack operation"], "B", "Internet connects many networks globally."),
    ("Section A Common Core", "Computer Networks", "LAN WAN MAN", "A network covering a city is generally called:", ["LAN", "MAN", "WAN", "PAN only"], "B", "MAN covers a metropolitan area."),
    ("Section A Common Core", "Computer Networks", "LAN WAN MAN", "A network spread across countries is generally:", ["LAN", "MAN", "WAN", "Star topology only"], "C", "WAN covers large geographic areas."),
    ("Section A Common Core", "Computer Networks", "network devices", "Which device regenerates a weak signal?", ["Repeater", "Router", "Database", "Stack"], "A", "A repeater regenerates signals."),
    ("Section A Common Core", "Computer Networks", "network devices", "A switch usually operates inside:", ["A local network", "A SQL query", "A postfix expression", "A pickle file"], "A", "Switches connect devices in a LAN."),
    ("Section A Common Core", "Computer Networks", "topologies", "Which topology has a single backbone cable?", ["Bus", "Star", "Ring", "Mesh"], "A", "Bus topology uses a common backbone."),
    ("Section A Common Core", "Computer Networks", "topologies", "In ring topology, each node is connected to:", ["A central hub only", "Two neighboring nodes", "Every other node", "No node"], "B", "Ring forms a closed loop of neighboring connections."),
    ("Section A Common Core", "Computer Networks", "MAC and IP address", "An IP address identifies:", ["A host/interface on a network", "A SQL column type", "Only a stack top", "A file exception"], "A", "IP addresses identify network hosts/interfaces."),
    ("Section A Common Core", "Computer Networks", "internet vs web", "Which one is broader?", ["World Wide Web", "Internet", "HTML page", "Browser tab only"], "B", "The web is a service on the internet, so internet is broader."),
    ("Section B1 Computer Science", "Exception and File Handling in Python", "syntax errors and exceptions", "A syntax error is detected mainly when:", ["Python parses invalid code", "A router forwards packets", "A SQL table is sorted", "A queue is full"], "A", "Syntax errors come from invalid program structure."),
    ("Section B1 Computer Science", "Exception and File Handling in Python", "syntax errors and exceptions", "Division by zero during program execution raises:", ["SyntaxError", "ZeroDivisionError", "NameError always", "EOFError always"], "B", "ZeroDivisionError occurs at runtime."),
    ("Section B1 Computer Science", "Exception and File Handling in Python", "try except else finally", "In Python, the else block of try-except runs when:", ["An exception occurs", "No exception occurs in try", "finally is absent", "The file is binary"], "B", "else runs only if try completes without exception."),
    ("Section B1 Computer Science", "Exception and File Handling in Python", "user-defined exceptions", "A user-defined exception class usually inherits from:", ["Exception", "list", "dict", "tuple"], "A", "Custom exception classes commonly inherit from Exception."),
    ("Section B1 Computer Science", "Exception and File Handling in Python", "raise exceptions", "raise ValueError('bad') will:", ["Handle the error silently", "Explicitly generate an exception", "Create a database", "Sort a list"], "B", "raise explicitly triggers an exception."),
    ("Section B1 Computer Science", "Exception and File Handling in Python", "text and binary files", "Binary files store data as:", ["Plain characters only", "Bytes", "SQL relations only", "Only integers"], "B", "Binary files store byte sequences."),
    ("Section B1 Computer Science", "Exception and File Handling in Python", "text and binary files", "Which method reads all lines of a text file into a list?", ["readlines()", "writelines() only", "pickle.dump()", "append()"], "A", "readlines returns file lines as a list."),
    ("Section B1 Computer Science", "Exception and File Handling in Python", "pickle", "Which pickle function writes an object to a binary file?", ["pickle.load()", "pickle.dump()", "open.read()", "file.close()"], "B", "pickle.dump serializes an object to a file."),
    ("Section B1 Computer Science", "Exception and File Handling in Python", "pickle", "Which pickle function reads a serialized object?", ["pickle.dump()", "pickle.load()", "pickle.write()", "pickle.fetch()"], "B", "pickle.load deserializes an object."),
    ("Section B1 Computer Science", "Exception and File Handling in Python", "file access modes", "Opening a file with mode 'a' means:", ["Read only", "Append", "Binary read", "Delete"], "B", "Mode a appends to the end of a file."),
    ("Section B1 Computer Science", "Stack", "LIFO", "If 10, 20, 30 are pushed into a stack, the first popped item is:", ["10", "20", "30", "None"], "C", "The last pushed item is popped first."),
    ("Section B1 Computer Science", "Stack", "push pop", "Removing the top element of a stack is called:", ["push", "pop", "enqueue", "dequeue"], "B", "pop removes the top stack element."),
    ("Section B1 Computer Science", "Stack", "list implementation", "In Python list implementation of stack, append() is commonly used for:", ["push", "pop", "peek only", "sort"], "A", "append adds an item to the top/end."),
    ("Section B1 Computer Science", "Stack", "prefix infix postfix", "A+B is an example of:", ["Prefix", "Infix", "Postfix", "Queue"], "B", "The operator appears between operands."),
    ("Section B1 Computer Science", "Stack", "expression evaluation", "Postfix expression 4 2 * 3 + evaluates to:", ["11", "14", "9", "24"], "A", "4*2+3 = 11."),
    ("Section B1 Computer Science", "Stack", "infix to postfix", "The postfix form of (A+B)*C is:", ["AB+C*", "ABC+*", "*+ABC", "A+BC*"], "A", "A+B becomes AB+, then multiply by C gives AB+C*."),
    ("Section B1 Computer Science", "Queue", "FIFO", "If 5, 6, 7 enter a queue, the first deleted item is:", ["5", "6", "7", "None"], "A", "Queue deletion follows first in first out."),
    ("Section B1 Computer Science", "Queue", "insert delete", "Deletion in a simple queue is generally from:", ["Rear", "Front", "Top", "Middle"], "B", "Queue deletion is from the front."),
    ("Section B1 Computer Science", "Queue", "list implementation", "In a Python list queue, inserting at rear can be done using:", ["append()", "pop() only", "sort()", "keys()"], "A", "append can add at the rear of a list-based queue."),
    ("Section B1 Computer Science", "Queue", "deque", "The deque class is available in:", ["collections", "math", "pickle only", "mysql"], "A", "collections.deque provides efficient queue operations."),
    ("Section B1 Computer Science", "Searching", "sequential search", "Sequential search compares the target with:", ["Each element one by one", "Only middle element", "Only sorted keys", "Only first and last always"], "A", "Linear search checks elements sequentially."),
    ("Section B1 Computer Science", "Searching", "binary search", "After one comparison, binary search usually:", ["Discards about half the search space", "Checks every item", "Sorts the array by itself", "Creates a table"], "A", "Binary search halves the search interval."),
    ("Section B1 Computer Science", "Searching", "best worst average case", "Best case of linear search occurs when the key is:", ["At the first position", "At the last position", "Absent", "Always sorted"], "A", "If first element matches, only one comparison is needed."),
    ("Section B1 Computer Science", "Searching", "python implementation", "In Python, list.index(x) returns:", ["The first index of x", "The sorted list", "The count of x only", "A binary file"], "A", "index returns the first matching index or raises an error if missing."),
    ("Section B1 Computer Science", "Sorting", "bubble sort", "In one pass of bubble sort ascending order, the largest element usually moves to:", ["Beginning", "End", "Middle only", "A queue"], "B", "Repeated adjacent swaps move the largest item to the end."),
    ("Section B1 Computer Science", "Sorting", "selection sort", "In selection sort ascending order, the first pass selects the:", ["Minimum element", "Median element", "Random key", "Last duplicate only"], "A", "Selection sort places the minimum at the first position."),
    ("Section B1 Computer Science", "Sorting", "insertion sort", "Insertion sort is efficient for:", ["Nearly sorted data", "Only network packets", "Only SQL joins", "Encrypted files only"], "A", "Insertion sort performs well on nearly sorted data."),
    ("Section B1 Computer Science", "Sorting", "dry run", "After first complete bubble-sort pass on [3, 1, 2] ascending, the list becomes:", ["[1, 2, 3]", "[1, 3, 2]", "[3, 2, 1]", "[2, 1, 3]"], "A", "Compare 3 and 1 -> [1,3,2], then 3 and 2 -> [1,2,3]."),
    ("Section B1 Computer Science", "Sorting", "dry run", "After one selection-sort pass on [4, 2, 5, 1], ascending order gives:", ["[1, 2, 5, 4]", "[4, 2, 5, 1]", "[2, 4, 5, 1]", "[5, 2, 4, 1]"], "A", "Minimum 1 is selected and swapped with the first element."),
    ("Section B1 Computer Science", "Sorting", "hashing", "Hashing is used for:", ["Fast storage and retrieval using a hash function", "Only text formatting", "Only creating networks", "Only exception handling"], "A", "Hashing maps keys to locations for faster access."),
    ("Section B1 Computer Science", "Sorting", "collision resolution", "Linear probing resolves collision by:", ["Checking next available slot", "Deleting the table", "Using a router", "Raising SyntaxError"], "A", "Linear probing searches the next slots sequentially."),
    ("Section B1 Computer Science", "Understanding Data", "data collection", "Primary data is collected:", ["First-hand from the source", "Only from old books", "Only by SQL COUNT", "Only by routers"], "A", "Primary data is collected directly."),
    ("Section B1 Computer Science", "Understanding Data", "data collection", "Data collected from existing reports is called:", ["Primary data", "Secondary data", "Stack data", "Packet data"], "B", "Secondary data comes from existing sources."),
    ("Section B1 Computer Science", "Understanding Data", "mean median", "Mean of 2, 4, 6 is:", ["4", "6", "12", "2"], "A", "Mean is (2+4+6)/3 = 4."),
    ("Section B1 Computer Science", "Understanding Data", "mean median", "Mode is the value that:", ["Occurs most frequently", "Is always middle", "Is total divided by count", "Is network speed"], "A", "Mode is the most frequent value."),
    ("Section B1 Computer Science", "Understanding Data", "standard deviation variance", "A lower standard deviation means data values are generally:", ["More spread out", "Closer to the mean", "Always zero", "Only text"], "B", "Low standard deviation means less spread."),
    ("Section B1 Computer Science", "Understanding Data", "data interpretation", "A bar chart is best for comparing:", ["Categories", "Only binary files", "Only stack top", "Only IP address"], "A", "Bar charts compare categorical values."),
    ("Section B1 Computer Science", "Data Communication and Security", "communication types", "Communication in both directions but not at the same time is:", ["Simplex", "Half-duplex", "Full-duplex", "Broadcast only"], "B", "Half-duplex allows both directions one at a time."),
    ("Section B1 Computer Science", "Data Communication and Security", "communication types", "Keyboard to CPU communication is commonly treated as:", ["Simplex", "Half-duplex", "Full-duplex", "Packet switching"], "A", "Simplex is one-way communication."),
    ("Section B1 Computer Science", "Data Communication and Security", "switching techniques", "Packet switching divides data into:", ["Packets", "Primary keys", "Stacks", "Tuples only"], "A", "Packet switching sends data in packets."),
    ("Section B1 Computer Science", "Data Communication and Security", "wired media", "Which is a guided transmission medium?", ["Twisted pair cable", "Radio wave", "Infrared", "Microwave"], "A", "Wired media are guided through physical cables."),
    ("Section B1 Computer Science", "Data Communication and Security", "wireless media", "Wi-Fi uses:", ["Wireless communication", "Twisted pair only", "Pickle only", "SQL only"], "A", "Wi-Fi is wireless networking."),
    ("Section B1 Computer Science", "Data Communication and Security", "protocols", "SMTP is mainly used for:", ["Email sending", "Web page transfer", "File sorting", "Stack pop"], "A", "SMTP is used for sending email."),
    ("Section B1 Computer Science", "Data Communication and Security", "bandwidth and data transfer rate", "Data transfer rate is measured in:", ["bits per second", "tuples per relation", "rows per key", "exceptions per file"], "A", "Transfer rate is commonly measured in bps."),
    ("Section B1 Computer Science", "Data Communication and Security", "viruses worms trojan spam cookies adware", "A trojan appears to be useful software but:", ["Performs harmful actions secretly", "Always sorts data", "Creates primary keys", "Is a SQL aggregate"], "A", "Trojans disguise malicious behavior."),
    ("Section B1 Computer Science", "Data Communication and Security", "viruses worms trojan spam cookies adware", "Unwanted bulk email is called:", ["Spam", "Cookie", "Firewall", "Tuple"], "A", "Spam refers to unwanted bulk messages."),
    ("Section B1 Computer Science", "Data Communication and Security", "firewall http https", "HTTP mainly operates at the:", ["Application layer", "Physical cable only", "Stack top", "Database schema"], "A", "HTTP is an application-layer protocol."),
    ("Section B1 Computer Science", "Data Communication and Security", "DoS intrusion snooping eavesdropping", "Eavesdropping means:", ["Secretly listening to communication", "Sorting a list", "Creating a table", "Opening binary file"], "A", "Eavesdropping is unauthorized listening/interception."),
]


def main() -> None:
    priority = load_priority()
    rows = []
    for index, item in enumerate(QUESTION_BANK, start=1):
        section, chapter, subtopic, text, options, correct, explanation = item
        match = priority[(priority["chapter"].eq(chapter)) & (priority["subtopic"].eq(subtopic))]
        priority_tier = match.iloc[0]["priority_tier"] if not match.empty else "Practice"
        raw_score = float(match.iloc[0]["raw_score"]) if not match.empty else 3.0
        rows.append(
            {
                "practice_id": stable_id(f"{chapter}-{subtopic}-{index}"),
                "section": section,
                "chapter": chapter,
                "subtopic": subtopic,
                "difficulty": difficulty(raw_score),
                "question_type": question_type(chapter, subtopic),
                "question_text": text,
                "option_a": options[0],
                "option_b": options[1],
                "option_c": options[2],
                "option_d": options[3],
                "correct_option": correct,
                "explanation": explanation,
                "priority_tier": priority_tier,
                "raw_score": raw_score,
                "source_basis": "CUET CS syllabus priority + imported paper structure; synthetic practice, not exact PYQ.",
            }
        )
    bank = pd.DataFrame(rows).sort_values(["raw_score", "chapter", "subtopic"], ascending=[False, True, True])
    bank.to_csv(PROCESSED_DIR / "practice_question_bank.csv", index=False)
    blueprint = pd.DataFrame(
        [
            {"mock_type": "quick", "questions": 10, "section_a": 4, "section_b1": 6, "use_case": "daily warmup"},
            {"mock_type": "focused", "questions": 20, "section_a": 7, "section_b1": 13, "use_case": "topic practice"},
            {"mock_type": "full_cuet_style", "questions": 40, "section_a": 15, "section_b1": 25, "use_case": "exam simulation"},
        ]
    )
    blueprint.to_csv(PROCESSED_DIR / "mock_blueprint.csv", index=False)
    print(f"Generated {len(bank)} CUET CS practice questions.")


def load_priority() -> pd.DataFrame:
    path = PROCESSED_DIR / "study_priority.csv"
    return pd.read_csv(path).fillna("") if path.exists() else pd.DataFrame()


def stable_id(value: str) -> str:
    return "mock_" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def difficulty(score: float) -> str:
    if score >= 4.6:
        return "medium"
    if score >= 3.6:
        return "easy-medium"
    return "easy"


def question_type(chapter: str, subtopic: str) -> str:
    text = f"{chapter} {subtopic}".lower()
    if "sql" in text or "join" in text or "group" in text:
        return "query-concept"
    if any(term in text for term in ["stack", "queue", "search", "sort", "postfix"]):
        return "dry-run"
    if "network" in text or "security" in text:
        return "concept-application"
    return "definition-based"


if __name__ == "__main__":
    main()
