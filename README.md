# SQL-Based Question Answering (QA) System

## Project Title

**Natural Language to SQL Question Answering System** using LangChain and LLMs.

---

## Overview

This project provides an **AI-powered QA system** that converts natural language questions into SQL queries, executes them on a relational database, and returns concise, human-readable results. Using the **Chinook database** as an example, it leverages **LangChain**, **LangGraph**, and **LLMs** for automated query generation, execution, and answer summarization.

---

## Objectives

* Allow users to query SQL databases using plain language.
* Automatically generate **syntactically correct, schema-aware SQL queries**.
* Summarize query results in natural, readable text.
* Handle ambiguous entities with embeddings for proper name resolution.

---

## Core Technologies

* **LLM:** Google Gemini 2.0 Flash
* **Frameworks:** LangChain, LangGraph
* **Database:** SQLite (`Chinook.db`)
* **Vector Store:** InMemoryVectorStore (for entity resolution)
* **Prompt Engineering:** LangChain `ChatPromptTemplate`
* **Environment Management:** `.env` file for API keys

---

## Key Features

1. **Natural Language Interface** – Converts user questions into SQL queries.
2. **Schema-Aware Query Generation** – Queries respect database schema.
3. **Modular Execution Pipeline** – *write-query → execute-query → generate-answer*.
4. **User Approval Checkpoint** – Optional pause before executing queries.
5. **Entity Resolution** – Embeddings normalize names and proper nouns.
6. **Visualization Support** – Mermaid diagrams to debug query flow.

---

## Challenges Addressed

* Mapping user intent to database structure.
* Avoiding unsafe or incorrect SQL execution.
* Resolving ambiguous entity names in queries.
* Creating reusable prompt templates for multiple databases.

---

## Outcomes

A **robust natural-language-to-SQL pipeline** with optional human oversight, suitable for analytics, database exploration, and research.

---

## Future Enhancements

* Support for production-grade databases (PostgreSQL, MySQL).
* Advanced SQL queries: joins, nested queries, multi-turn questions.
* Improved caching and result summarization.

---

## Installation

```bash
git clone https://github.com/yourusername/sql-qa-agent.git
cd sql-qa-agent
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_api_key_here
```

---

## Usage

### CLI Mode

```bash
python llm.py
```

### Example Interaction

```
User: "How many albums does AC/DC have?"
Agent → Generates SQL → Executes Query → Returns: "AC/DC has 10 albums."
```

---

## License

MIT License – free to use, modify, and distribute.
