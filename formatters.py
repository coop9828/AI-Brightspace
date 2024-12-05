import re
import logging
from typing import Dict, List, Optional


def clean_llm_response_question(response):
    """
    Extract the discussion question from the LLM response.

    Args:
        response (str): Raw response from the LLM

    Returns:
        str: Cleaned discussion question
    """
    try:
        # Parse the response as a dictionary if it's a string
        if isinstance(response, str):
            response = eval(response)

        # Extract content from the assistant's message
        content = response['content']

        # Find the question after "Discussion Question:" using regex
        match = re.search(r'\*\*Discussion Question:\*\* (.*?)(?=\n|$)', content)
        if match:
            return match.group(1).strip()

        # Fallback: return everything after "Discussion Question:"
        if "Discussion Question:" in content:
            return content.split("Discussion Question:")[-1].strip()

        return content.strip()

    except Exception as e:
        print(f"Error cleaning response: {e}")
        return response


def clean_llm_response_requirements(response):
    """
    Extract and clean requirements from the LLM response, ensuring proper format for grading.
    """
    try:
        # Find the assistant's content with the requirements
        if isinstance(response, list):
            for item in response:
                if isinstance(item, dict) and item.get('role') == 'assistant':
                    content = item.get('content', '')
                    if '1.' in content and '2.' in content and '3.' in content:
                        # Extract the numbered requirements
                        lines = content.split('\n')
                        requirements = []
                        current_req = []

                        for line in lines:
                            if re.match(r'^\d+\.', line.strip()):
                                if current_req:
                                    requirements.append(' '.join(current_req))
                                    current_req = []
                                current_req.append(line.strip())
                            elif line.strip() and current_req:
                                current_req.append(line.strip())

                        if current_req:
                            requirements.append(' '.join(current_req))

                        return '\n'.join(requirements)

        # Fallback: try to extract any numbered list format
        if isinstance(response, str):
            content = response
        else:
            content = str(response)

        if '1.' in content and '2.' in content and '3.' in content:
            # Extract numbered requirements
            pattern = r'\d+\.\s+[^\d].*?(?=\d+\.|$)'
            requirements = re.findall(pattern, content, re.DOTALL)
            if requirements:
                return '\n'.join(req.strip() for req in requirements)

        return content.strip()

    except Exception as e:
        print(f"Error cleaning requirements: {e}")
        return str(response)

def clean_llm_response_feedback(response):
    """
    Extract and clean the information from the LLM response containing
    'Missing Concepts:' and 'Suggestions'.

    Args:
        response (str): Raw response from the LLM

    Returns:
        str: Cleaned response
    """
    try:
        # Find the assistant's response
        assistant_response = next((entry['content'] for entry in response if entry['role'] == 'assistant'), "")

        # Initialize cleaned content
        cleaned_content = {}

        # Extract "Missing Concepts" section
        if "Missing Concepts:" in assistant_response:
            missing_concepts = assistant_response.split("Missing Concepts:")[-1]
            missing_concepts = missing_concepts.split("Suggestions:")[0].strip()
            cleaned_content['Missing Concepts'] = missing_concepts

        # Extract "Suggestions" section
        if "Suggestions:" in assistant_response:
            suggestions = assistant_response.split("Suggestions:")[-1].strip()
            cleaned_content['Suggestions'] = suggestions

        return cleaned_content

    except Exception as e:
        print(f"Error cleaning response: {e}")
        return {}

def create_questions(pipe, text):
    """Generate discussion question from text with error handling"""
    try:
        messages = [
            {"role": "system", "content": "You are a teaching assistant tasked with creating discussion questions."},
            {"role": "user", "content": f"Create one discussion question from this text: {text}. Denote the discussion question by 'Discussion Question:'"}
        ]
        outputs = pipe(messages, max_new_tokens=1024)

        if not outputs or not outputs[0]["generated_text"]:
            raise ValueError("No question generated")

        question = outputs[0]["generated_text"][-1]
        cleaned_q = clean_llm_response_question(question)

        if not cleaned_q:
            raise ValueError("Question cleaning resulted in empty content")

        return cleaned_q

    except Exception as e:
        logging.error(f"Error generating question: {e}")
        raise

def create_requirements(pipe, text, cleaned_question):
    """Generate requirements with proper error handling"""
    try:
        messages = [
            {"role": "system", "content": """You are a teaching assistant creating grading requirements.
            Create 3-4 clear requirements that focus on the main concepts and content rather than technical writing elements.
            Each requirement should:
            1. Focus on understanding and analysis of the core content
            2. Be clearly measurable
            3. Have a specific point value
            4. Total exactly 100 points

            Format as:
            1. [Main concept requirement] (X points): [Brief description of what's expected]
            'X points' must total exactly 100 points.
            Avoid:
            - Repetitive requirements
            - Overly technical writing criteria
            - More than 4 requirements
            - Breaking down points into sub-categories"""},
            {"role": "user", "content": f"Create specific, measurable requirements for grading responses to:\n\n{cleaned_question}\n\nBased on:\n\n{text}"}
        ]

        outputs = pipe(messages, max_new_tokens=1024)
        if not outputs or not outputs[0]["generated_text"]:
            raise ValueError("No requirements generated")

        raw_requirements = outputs[0]["generated_text"]
        cleaned_requirements = clean_llm_response_requirements(raw_requirements)

        if not cleaned_requirements:
            raise ValueError("Requirements cleaning resulted in empty content")

        return cleaned_requirements

    except Exception as e:
        logging.error(f"Error generating requirements: {e}")
        raise

def display_feedback(feedback_data):
    """Enhanced feedback display with better error handling"""
    try:
        print("\n=== Feedback Summary ===")
        print(f"Overall Score: {feedback_data.get('composite_score', 0):.1f}%")

        automated_grading = feedback_data.get('automated_grading', {})
        print(f"Automated Score: {automated_grading.get('percentage', 0):.1f}%")
        print(f"Total Points: {automated_grading.get('total_points', 0)}/{automated_grading.get('max_points', 100)}")

        print("\n=== LLM Feedback ===")
        llm_feedback = feedback_data.get('llm_feedback', {})

        if llm_feedback.get('missing_concepts'):
            print("\nMissing Concepts:")
            for concept in llm_feedback['missing_concepts']:
                print(f"  - {concept}")

        if llm_feedback.get('areas_of_strength'):
            print("\nAreas of Strength:")
            for strength in llm_feedback['areas_of_strength']:
                print(f"  - {strength}")

        if llm_feedback.get('suggestions'):
            print("\nSuggestions for Improvement:")
            for suggestion in llm_feedback['suggestions']:
                print(f"  - {suggestion}")

        print("\n=== Detailed Requirement Feedback ===")
        for item in automated_grading.get('feedback', []):
            print(f"\n{item.get('status', '?')} Requirement: {item.get('requirement', 'N/A')}")
            print(f"Score: {item.get('points_earned', 0)}/{item.get('max_points', 0)} points")

            if item.get('feedback'):
                print("\nFeedback:")
                for feedback_item in item['feedback']:
                    if isinstance(feedback_item, dict):
                        for key, value in feedback_item.items():
                            print(f"  {key}: {value}")
                    else:
                        print(f"  - {feedback_item}")

    except Exception as e:
        logging.error(f"Error displaying feedback: {e}")
        print("\nError displaying feedback. Please check the log for details.")