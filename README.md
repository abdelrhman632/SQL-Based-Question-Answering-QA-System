**Project Report: SQL-Based Question Answering (QA) System**

**Project Title:** Natural Language to SQL Question Answering System using LangChain and LLMs

**Overview:**
This project presents an AI-powered Question Answering (QA) system capable of translating natural language queries into SQL commands, executing them on a relational database, and returning the results in a user-friendly format. The system was developed using the Chinook database, Google Gemini LLM, LangChain, LangGraph, and associated tools.

**Objectives:**

* Enable users to interact with SQL databases using natural language.
* Automatically generate and execute syntactically valid SQL queries.
* Summarize SQL outputs into clear and concise answers.
* Ensure query safety and allow human-in-the-loop decision-making.

**Core Technologies:**

* **LLM:** Google Gemini 2.0 Flash
* **Frameworks:** LangChain, LangGraph
* **Database:** SQLite (Chinook.db)
* **Prompt Engineering:** LangChain ChatPromptTemplate with structured outputs
* **Tools:** SQLDatabaseToolkit, QuerySQLDatabaseTool, InMemoryVectorStore

**Key Features:**

1. **Natural Language Interface:** Accepts user questions and uses prompt templates to generate SQL queries.
2. **Schema-Aware Query Generation:** Integrates real database schema to ensure valid query creation.
3. **Execution Pipeline with LangGraph:** Modular workflow with interrupt points, structured in write-query → execute-query → generate-answer sequence.
4. **User Approval Checkpoint:** Queries can be paused before execution for manual review.
5. **Result Summarization:** Converts raw SQL results into natural language responses.
6. **Retriever Tool for Entity Resolution:** Uses vector embeddings to correct and normalize proper nouns.
7. **Visualization:** Generates Mermaid diagrams for graph debugging.

**Challenges Addressed:**

* Misalignment between user intent and database schema.
* Incorrect or unsafe queries.
* Ambiguity in proper nouns (e.g., artist names, city names).
* Scalability and reusability of prompt templates across different databases.

**Outcome:**
The system successfully demonstrates a robust and interactive natural-language-to-SQL pipeline with human intervention capabilities. It’s well-suited for enterprise database applications, analytics platforms, or internal tools requiring natural language access to structured data.

**Future Work:**

* Integration with live databases (e.g., PostgreSQL, MySQL).
* Adding feedback loop for active learning and self-improvement.
* Enhancing security through query validation rules and sandboxing.
* Expanding support for joins, nested queries, and multi-turn dialogue.

