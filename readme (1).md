# Auto Discussion Question Generator and Grader

An AI-powered system that automatically generates discussion questions from educational content, creates grading requirements, and evaluates student responses with detailed feedback. This tool aims to assist educators in creating engaging discussion questions and providing consistent, detailed feedback to students.

[Insert System Architecture Diagram Here - showing flow from input text → question generation → grading requirements → student response → feedback]

## 🌟 Features

- **Intelligent Question Generation**: Automatically creates relevant discussion questions from educational content
- **Automated Grading Framework**: 
  - Creates specific grading requirements with point allocations
  - Weighted scoring system
  - Concept coverage analysis
- **Comprehensive Feedback System**:
  - Missing concepts identification
  - Areas of strength analysis
  - Actionable improvement suggestions
  - Numerical scoring with detailed breakdowns

[Insert Screenshot of Example Output Here - showing generated question, requirements, and feedback]

## 🏗️ Project Structure

```
├── main_app.py       # Main application logic and LLM pipeline
├── formatters.py     # Response cleaning and text formatting utilities
├── grader.py         # Automated grading system implementation
└── requirements.txt  # Project dependencies
```

## 📋 Requirements

### Core Dependencies
```txt
torch>=2.0.0
transformers>=4.30.0
huggingface_hub>=0.16.0
spacy>=3.5.0
tensorflow>=2.13.0
scikit-learn>=1.3.0
```

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- 8GB+ RAM
- Internet connection for model downloads

## 🚀 Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/auto-discussion-grader.git
cd auto-discussion-grader
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Configure Hugging Face access:
```python
# Set your Hugging Face API token
export HUGGINGFACE_TOKEN='your_token_here'
```

[Insert Screenshot of Successful Setup Here]

## 💡 Usage

### Basic Usage
```bash
python main_app.py --input_file path/to/lecture.txt
```

### Advanced Configuration
```bash
python main_app.py --input_file path/to/lecture.txt --model meta-llama/Llama-3.2-1B-Instruct --threshold 0.6
```

[Insert GIF/Video of Usage Example Here]

## 🔍 Components

### AutoGrader
- NLP-based response analysis
- Concept coverage calculation
- Semantic similarity matching
- Response depth evaluation

### AnswerAssessment
- Hybrid LLM-automated grading
- Composite scoring system
- Detailed feedback generation
- Performance analytics

[Insert Component Interaction Diagram Here]

## ⚠️ Limitations

- Feedback quality depends on input text clarity
- Limited to text-based educational content
- Requires parameter tuning for different subjects
- LLM responses need occasional verification
- Processing speed depends on hardware capabilities

## 🔮 Future Improvements

- LMS integration (Canvas, Blackboard, Moodle)
- Enhanced instructor feedback loop
- Multiple question type support
- Domain-specific training capabilities
- Real-time grading optimization

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Related Resources

### AI in Education
- [Carnegie Learning's AI Research](https://www.carnegielearning.com/research/)
- [EdX's AI in Education Course](https://www.edx.org/learn/artificial-intelligence)
- [Stanford's AI in Education Research](https://ai.stanford.edu/research/ai-education)

### Technical Resources
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [spaCy Course](https://course.spacy.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

### Research Papers
1. ["AI in Education: Current Trends and Future Perspectives"](https://www.frontiersin.org/articles/10.3389/frai.2022.871043/)
2. ["Automated Essay Scoring and the Future of Educational Assessment"](https://eric.ed.gov/?id=EJ1067807)
3. ["Natural Language Processing in Education"](https://www.sciencedirect.com/science/article/pii/S2666920X21000157)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- Meta AI for the LLaMA model
- Hugging Face for transformer implementations
- spaCy for NLP tools
- The educational technology community

[Insert Project Team/Contributors Photos Here]
