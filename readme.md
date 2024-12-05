# Auto Discussion Question Generator and Grader

An AI-powered system that automatically generates discussion questions from educational content, creates grading requirements, and evaluates student responses with detailed feedback.

## Features

- Generates relevant discussion questions from input text
- Creates specific grading requirements with point allocations
- Evaluates student responses using both LLM and automated grading
- Provides detailed feedback with:
  - Missing concepts identification
  - Areas of strength
  - Suggestions for improvement
  - Numerical scoring

## Project Structure

```
├── main_app.py       # Main application logic
├── formatters.py     # Response cleaning utilities
└── grader.py         # Grading system implementation
```

## Requirements

- Python 3.8+
- huggingface_hub
- transformers
- scikit-learn
- spacy
- torch
- en_core_web_sm (spaCy model)

## Setup

1. Install dependencies:
```bash
pip install huggingface_hub transformers scikit-learn spacy torch
python -m spacy download en_core_web_sm
```

2. Configure Hugging Face access:
   - Get an API token from Hugging Face
   - Set it as an environment variable or in your configuration

## Usage

1. Prepare your lecture content in a text file
2. Run the application:
```bash
python main_app.py
```
3. The system will:
   - Generate a discussion question
   - Create grading requirements
   - Accept student responses
   - Provide detailed feedback

## Components

### AutoGrader
- Analyzes responses using NLP techniques
- Calculates concept coverage and response depth
- Generates detailed feedback on missing concepts

### AnswerAssessment
- Combines LLM and automated grading approaches
- Provides composite scoring
- Generates actionable feedback

## Limitations

- Feedback generation quality may vary without specific training data
- Limited to text-based educational content
- Requires careful tuning of grading parameters
- LLM-based feedback may need human verification

## Future Improvements

- Integration with learning management systems
- Enhanced feedback loop for instructors
- Support for multiple question types
- Training on domain-specific educational data

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

[Insert your chosen license here]
