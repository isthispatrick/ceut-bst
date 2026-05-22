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
