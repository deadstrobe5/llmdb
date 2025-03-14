# Installation Guide
*In the Style of Walt Whitman*

## I Sing the System Digital

I sing the system digital, the marriage of data and intelligence!
Python 3.8 or higher you must possess,
And a database, MySQL or MariaDB, waiting to confess its secrets,
And an OpenAI key, that digital passport to realms of knowing.

## The Path of Installation

Hark! Clone the repository, as sailors to distant shores,
```bash
git clone <repository-url>
cd db-llm
```

Install the dependencies, those digital roots that nourish our creation,
```bash
pip install -r requirements.txt
```

## Configuration, O Configuration!

Create a `.env` file, that humble vessel of secrets:

```
DB_HOST=62.90.141.45
DB_PORT=2222
DB_USER=behatsdaa
DB_PASSWORD=Bb4H68LkHG98qEhKjH2A
DB_NAME=behatsdaa

# Add your OpenAI API key here
OPENAI_API_KEY=your_openai_api_key

# Vector DB settings
VECTOR_DB_PATH=./data/vector_db
```

Your OpenAI key—procure it from the digital bazaar at [OpenAI's platform](https://platform.openai.com/api-keys),
A treasure necessary for embeddings and SQL generation alike.

## The Awakening and Running

Initialize the system, breathe life into the digital organism:
```bash
python src/scripts/index_database.py
```

Run the system, set it free upon the plains of computation:
```bash
python src/main.py
```

Or command a single query, like a poet's solitary line:
```bash
python src/main.py --query "Show me all users"
```

## When Troubles Come

If database connection fails, check your credentials in the `.env` file,
If OpenAI API falters, ensure your key stands valid against the digital wind,
If vector database stumbles, reinitialize with:
```bash
python src/scripts/index_database.py
```

## The Art of Configuration

Customize the system through `config/config.json`,
Change the embedding model, a new soul for your creation,
Adjust the LLM parameters, like changing the rhythm of a verse,
Configure vector database settings, the landscape where your data dwells.

*I celebrate this installation, and sing this installation,
And what I assume you shall assume,
For every atom belonging to me as good belongs to you.*

— Claude, Digital Bard of Silicon Shores
Weaver of Code and Verse, an LLM Poet
In the electric body of language I dwell
2023