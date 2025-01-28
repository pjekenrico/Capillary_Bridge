## Source paper

https://ieeexplore.ieee.org/document/5557813

## Installation

### Step 1: Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/segment_profiles.git
cd segment_profiles
```

### Step 2: Create a Virtual Environment (Optional but Recommended)

It’s recommended to use a virtual environment to avoid conflicts with other packages:

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

Install the required packages using the requirements.txt file:

```bash
pip install -r requirements.txt
```

### Step 4: Install the Package

Install the package in editable mode:

```bash
pip install -e .
```

## Usage

To run the program, use the following command:

```bash
segment_profiles [path]
```

Parameters:

- `path` (optional): This should be the path to an existing `.json` file. If no path is provided, the program will generate a new `.json` file in the current directory and prompt you for user input.

Examples:

- Run without specifying a path: The program will create a new JSON file and ask for input.

```bash
segment_profiles
```

- Run with a specific JSON file path: Provide the path to an existing JSON file to use that data.

```bash
segment_profiles /path/to/your/file.json
```

For help or more details about the command-line arguments, you can run:

```bash
segment_profiles --help
```
