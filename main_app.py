import logging
import torch
from transformers import pipeline
from formatters import clean_llm_response_question, clean_llm_response_requirements, create_questions, create_requirements, display_feedback
from grader import AnswerAssessment, ImprovedAutoGrader
from huggingface_hub import login,whoami

key = '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'

if key:
    login(key)

    # Verify the login status
    try:
        user_info = whoami()  # Get user information
        print(f"Successfully logged in as {user_info['name']}.")
    except Exception as e:
        print("Login failed. Please check your Hugging Face token.")
        print(f"Error: {e}")
else:
    print("No Hugging Face token provided.")
class AutoGrader:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),  # Include phrases up to 3 words
            max_features=5000,
            min_df = 1,
            strip_accents='unicode'

        )
    def _preprocess_response(self, response):
        """Add preprocessing to normalize response format"""
        # Remove extra whitespace and normalize spaces
        response = ' '.join(response.split())

        # Handle common formatting issues
        response = response.replace('  ', ' ')
        response = re.sub(r'\s+([.,!?])', r'\1', response)

        # Remove any markdown formatting that might interfere with grading
        response = re.sub(r'[#*_~`]', '', response)

        # Standardize newlines
        response = response.replace('\r\n', '\n')

        # Remove any URLs that might be present
        response = re.sub(r'http\S+|www.\S+', '', response)

        return response
    def parse_requirements(self, requirements_text):
        """Parse requirements with robust format handling"""
        requirements_text = self._preprocess_response(requirements_text)

        requirements = []
        lines = requirements_text.strip().split('\n')
        current_requirement = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for new requirement (starts with number)
            if re.match(r'^\d+\.', line):
                if current_requirement:
                    requirements.append(current_requirement)

                # Extract points
                points_match = re.search(r'\((\d+)\s*points?\)', line)
                points = int(points_match.group(1)) if points_match else 30

                # Clean the content
                content = re.sub(r'\((\d+)\s*points?\)', '', line)  # Remove points
                content = re.sub(r'^\d+\.\s*', '', content)  # Remove number
                content = content.strip(': ')  # Remove extra characters

                current_requirement = {
                    'content': content,
                    'points': points,
                    'key_concepts': self._extract_key_concepts(content)
                }
            elif current_requirement:
                # Append to current requirement content
                current_requirement['content'] += ' ' + line
                current_requirement['key_concepts'].extend(self._extract_key_concepts(line))

        # Add final requirement
        if current_requirement:
            requirements.append(current_requirement)

        return requirements

    def _extract_key_concepts(self, text):
        """Extract key concepts using improved NLP analysis"""
        doc = self.nlp(text)
        concepts = []

        # Extract noun phrases and named entities
        for chunk in doc.noun_chunks:
            concepts.append(chunk.text.lower())

        for ent in doc.ents:
            concepts.append(ent.text.lower())

        # Extract important verb phrases
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                phrase = self._get_verb_phrase(token)
                if phrase:
                    concepts.append(phrase.lower())

        # Clean and deduplicate concepts
        concepts = [c.strip() for c in concepts if len(c.strip()) > 3]
        return list(set(concepts))

    def _get_verb_phrase(self, verb_token):
        """Extract meaningful verb phrases with their objects"""
        phrase_parts = [verb_token.text]

        for child in verb_token.children:
            if child.dep_ in ["dobj", "pobj", "attr"]:
                phrase_parts.extend([t.text for t in child.subtree])

        return " ".join(phrase_parts)

    def grade_response(self, requirements, student_response):
        """Enhanced grading with more nuanced scoring and feedback"""
        total_points = 0
        max_points = 0
        feedback = []

        student_doc = self.nlp(student_response)
        student_concepts = set(self._extract_key_concepts(student_response))

        # Calculate global coherence
        global_relevance = self._calculate_similarity(
            ' '.join(req['content'] for req in requirements),
            student_response
        )

        for req in requirements:
            max_points += req['points']

            # Calculate scores with adjusted weights
            concept_coverage = self._calculate_concept_coverage(req['key_concepts'], student_concepts)
            content_relevance = self._calculate_similarity(req['content'], student_response)
            depth_score = self._analyze_response_depth(student_response, req['content'])

            # Adjust weights based on requirement importance
            weights = {
                'concept': 0.35,
                'relevance': 0.25,
                'depth': 0.25,
                'coherence': 0.15
            }

            # Calculate weighted score
            score = (
                concept_coverage * weights['concept'] +
                content_relevance * weights['relevance'] +
                depth_score * weights['depth'] +
                global_relevance * weights['coherence']
            )

            # Apply minimum score threshold
            score = max(0.3, score)  # Ensure minimum 20% score if response attempts the requirement

            points_earned = round(score * req['points'])
            total_points += points_earned

            # Generate feedback
            feedback.append({
                'requirement': req['content'],
                'points_earned': points_earned,
                'max_points': req['points'],
                'status': "✓" if points_earned >= 0.7 * req['points'] else "×",
                'missing_concepts': list(set(req['key_concepts']) - student_concepts),
                'feedback': self._generate_requirement_feedback(
                    score, concept_coverage, content_relevance, depth_score
                )
            })

        return {
            'total_points': total_points,
            'max_points': max_points,
            'percentage': (total_points / max_points * 100),
            'feedback': feedback,
            'summary': {
                'strengths': [f for f in feedback if 'positive' in f],
                'improvements_needed': [f for f in feedback if 'issue' in f],
                'suggestions': [f['suggestion'] for f in feedback if 'suggestion' in f]
            },
            'rubric_breakdown': feedback
        }

    def _analyze_response_depth(self, response, requirement):
        """Enhanced depth analysis considering multiple factors"""
        doc = self.nlp(response)

        # Analyze relevant sentences with improved relevance threshold
        relevant_sents = [sent for sent in doc.sents
                         if self._calculate_similarity(sent.text, requirement) > 0.2]

        if not relevant_sents:
            return 0.0

        # Calculate multiple metrics
        avg_sent_length = np.mean([len(sent) for sent in relevant_sents])
        num_entities = len([ent for sent in relevant_sents for ent in sent.ents])

        # Analysis of sentence complexity
        complexity_scores = []
        for sent in relevant_sents:
            # Count subordinate clauses
            num_subordinate = len([token for token in sent
                                 if token.dep_ in ['advcl', 'ccomp', 'xcomp']])
            # Count logical connectors
            num_connectors = len([token for token in sent
                                if token.dep_ in ['cc', 'mark']])

            complexity = (1 + num_subordinate + 0.5 * num_connectors) / len(sent)
            complexity_scores.append(complexity)

        avg_complexity = np.mean(complexity_scores) if complexity_scores else 0

        # Combine metrics with weights
        depth_score = (
            0.25 * min(1.0, len(relevant_sents) / 8) +    # Number of relevant sentences
            0.20 * min(1.0, avg_sent_length / 20) +       # Average sentence length
            0.20 * min(1.0, num_entities / 8) +           # Named entity usage
            0.20 * min(1.0, avg_complexity) +             # Sentence complexity
            0.15 * paragraph_score                        # Paragraph structure
        )

        return depth_score

    def _generate_requirement_feedback(self, overall_score, concept_score, relevance_score, depth_score):
        """Generate more specific and actionable feedback"""
        feedback = []

        # Concept coverage feedback
        if concept_score < 0.7:
          missing = "Many" if concept_score < 0.4 else "Some"
          feedback.append({
              'area': 'Key Concepts',
              'issue': f"{missing} key concepts could be better addressed",
              'suggestion': "Try to explicitly discuss concepts like [specific concepts]. Consider using course terminology more directly."
          })

        # Relevance feedback
        if relevance_score < 0.7:
            if relevance_score < 0.4:
                feedback.append("Response could be more focused on the specific requirement")
            else:
                feedback.append("Could improve alignment with the requirement")

        # Depth feedback
        if depth_score < 0.7:
            if depth_score < 0.4:
              feedback.append({
                  'area': 'Analysis Depth',
                  'issue': "Response lacks detailed analysis",
                  'suggestion': "Expand your response with specific examples and more detailed explanations of key points. Add supporting evidence for your claims."
              })
            else:
                feedback.append("Could expand on some points with more detail")
        strengths = []
        if concept_score >= 0.7:
            strengths.append("Good use of key concepts")
        if relevance_score >= 0.7:
            strengths.append("Strong relevance to the topic")

        if strengths:
            feedback.append({
                'area': 'Strengths',
                'positive': ', '.join(strengths)
            })
        # Add positive feedback for good scores
        if all(score >= 0.7 for score in [concept_score, relevance_score, depth_score]):
            feedback.append("Strong, well-developed response that addresses the requirement effectively")
        elif any(score >= 0.8 for score in [concept_score, relevance_score, depth_score]):
            feedback.append("Shows good understanding in some areas")

        return feedback

    def _calculate_similarity(self, text1, text2):
        """Calculate semantic similarity with improved error handling"""
        try:
            # Ensure non-empty input
            if not text1.strip() or not text2.strip():
                return 0.0

            # Generate n-gram features with error handling
            texts = [text1.lower(), text2.lower()]

            try:
                tfidf_matrix = self.vectorizer.fit_transform(texts)
                if tfidf_matrix.shape[1] == 0:  # Check if vocabulary is empty
                    return self._fallback_similarity(text1, text2)
                base_similarity = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
            except (ValueError, NotFittedError) as e:
                logging.warning(f"TF-IDF calculation failed: {e}")
                return self._fallback_similarity(text1, text2)

            # Add weight for key phrase matches
            doc1 = self.nlp(text1.lower())
            doc2 = self.nlp(text2.lower())

            # Extract key phrases (noun chunks and named entities)
            phrases1 = set([chunk.text for chunk in doc1.noun_chunks] + [ent.text for ent in doc1.ents])
            phrases2 = set([chunk.text for chunk in doc2.noun_chunks] + [ent.text for ent in doc2.ents])

            # Calculate phrase overlap with error handling
            phrase_overlap = len(phrases1.intersection(phrases2)) / max(len(phrases1), 1) if phrases1 else 0

            # Weighted combination
            return 0.7 * base_similarity + 0.3 * phrase_overlap

        except Exception as e:
            logging.error(f"Error in similarity calculation: {e}")
            return self._fallback_similarity(text1, text2)

    def _fallback_similarity(self, text1, text2):
        """Simple fallback similarity measure"""
        # Convert to sets of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _calculate_concept_coverage(self, required_concepts, student_concepts):
        """Calculate concept coverage with weighted importance"""
        if not required_concepts:
            return 1.0

        matches = len(set(required_concepts) & student_concepts)
        return matches / len(required_concepts)

def main():
    try:
        # Initialize components
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        answer_assessment = AnswerAssessment(pipe)
        answer_assessment.grader = ImprovedAutoGrader()

        # Get input text

        input_file = '/Users/cooperchristian/Library/CloudStorage/OneDrive-Vanderbilt/DataScience/Second Year/Fall/Transformers/Final Project/test_files/sample_lecture.txt'
        with open(input_file, 'r', encoding='utf-8') as file:
          source_text = file.read()
        print(len(source_text))
        if not source_text:
            raise ValueError("Empty source text provided")

        # Generate question
        question = create_questions(pipe, source_text)
        print("\nGenerated Question:")
        print(question)

        # Generate requirements
        requirements = create_requirements(pipe, source_text, question)
        print("\nGrading Requirements:")
        print(requirements)

        # Get student response
        print("\nProvide Response:")
        student_answer = input().strip()

        if not student_answer:
            raise ValueError("Empty student answer provided")

        # Process feedback
        feedback = answer_assessment.assess_answer(
            question=question,
            correct_concepts=requirements,
            student_answer=student_answer
        )

        # Display results
        display_feedback(feedback)

    except Exception as e:
        print(f"Error in execution: {str(e)}")
        logging.error(f"Detailed error: {e}", exc_info=True)