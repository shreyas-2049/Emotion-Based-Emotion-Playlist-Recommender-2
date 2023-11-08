# Mood-Based Song Recommendation System using Deep Learning and OpenCV

## Overview

This project combines advanced Deep Learning techniques, specifically Convolutional Neural Networks (CNNs) implemented with the KERAS library, and OpenCV for real-time facial expression recognition. The system accurately categorizes facial expressions into emotions such as Happy, Sad, Neutral, Surprise, etc., enhancing user interaction and personalization. The project also integrates a sophisticated Music Recommendation System using the XGBOOST algorithm, providing users with personalized song suggestions based on their detected emotions.

## How it Works

### Facial Expression Recognition:

1. **Deep Learning with CNNs:**
   - Utilizes Convolutional Neural Networks (CNNs) implemented using the KERAS library.
   - The model is trained on the FER-2013 dataset, comprising grayscale facial images of various expressions.
  
 ![image](https://github.com/Ritik-Bhasarkar/Music-Recommendation-using-Facial-Emotion/assets/71097818/c9651b2e-1195-43aa-9c1c-029c296be840)


2. **OpenCV Integration:**
   - Utilizes OpenCV for real-time face detection through Haar Cascade classifiers.
   - Captures facial expressions through the webcam, processes the images, and feeds them into the CNN model.

3. **Expression Tagging:**
   - The system recognizes human expressions, tagging them as Happy, Sad, Neutral, Surprise, etc.
   - Accuracy and robustness are achieved through the deep learning model's training on diverse facial expressions.

### Music Recommendation:

1. **XGBOOST Algorithm:**
   - Implements the XGBOOST algorithm for personalized music recommendations.
   - The detected facial expression serves as input, and the model suggests songs tailored to the user's mood.

2. **Spotify Integration (Optional):**
   - Utilizes Spotify's Web API for retrieving song recommendations based on the recognized emotion.
   - Enhances user experience by providing a diverse and personalized playlist.

## Dataset

The FER-2013 dataset forms the backbone of the facial expression recognition model. It consists of 48x48 pixel grayscale images representing various facial expressions, enabling the CNN model to learn and categorize emotions accurately.

## Graphs and Human Detection

![Human Detection](https://github.com/Ritik-Bhasarkar/Music-Recommendation-using-Facial-Emotion/assets/71097818/bc61d12d-a372-4fbe-b81d-6c1d5f6c9870)




![Accuracy graph](https://github.com/Ritik-Bhasarkar/Music-Recommendation-using-Facial-Emotion/assets/71097818/03b18a9a-d9d6-4605-b1ac-c2a0f1921517)

For a visual representation of the system's performance, refer to the provided graphs showcasing accuracy, loss, or any relevant metrics. Additionally, human detection photos will demonstrate the system in action, capturing facial expressions for accurate emotion recognition.

## Usage

1. **Run the Application:**
   - Execute the main script, e.g., `python main.py`.
   - Grant webcam access permissions for facial expression detection.

2. **Experience Personalized Music:**
   - The system will detect and recognize your facial expression.
   - Based on the detected emotion, enjoy personalized music recommendations tailored to your mood.

## Disclaimer

This project respects user privacy and does not store or transmit any personal data. It solely processes facial expressions locally for mood recognition. Users are encouraged to use the system responsibly and be aware of privacy and consent considerations.


