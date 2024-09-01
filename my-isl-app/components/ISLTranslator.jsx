import * as tf from '@tensorflow/tfjs';
import { useEffect, useState } from "react";

const ISLTranslator = () => {
    const [model, setModel] = useState(null);
    const [prediction, setPrediction] = useState(null);

    const loadModel = async () => {
        const loadedModel = await tf.loadGraphModel('/models2/model.json');
        setModel(loadedModel);
    };

    useEffect(() => {
        loadModel();
    }, []);

    const preprocessInput = (imageElement) => {
        // Convert image to tensor and preprocess
        const imageTensor = tf.browser.fromPixels(imageElement);
        const resized = tf.image.resizeBilinear(imageTensor, [64, 64]); // Adjust size based on your model's input
        const normalized = resized.div(tf.scalar(255)); // Normalize pixel values
        const input = normalized.expandDims(0); // Add batch dimension
        return input;
    };

    const detect = async () => {
        if (model) {
            const img = document.getElementById('test-image'); // Get the image element
            const input = preprocessInput(img); // Preprocess the image
            const predictions = await model.predict(input); // Make prediction

            const predictedIndex = predictions.argMax(-1).dataSync()[0]; // Get predicted class index
            const classLabels = [
                '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                'U', 'V', 'W', 'X', 'Y', 'Z'
            ];
            setPrediction(classLabels[predictedIndex]); // Set the prediction
            console.log("Predicted Sign: ", classLabels[predictedIndex]); // Log the prediction
        }
    };

    useEffect(() => {
        detect(); // Run detection once when the model is loaded
    }, [model]);

    return (
        <div>
            <img id="test-image" src="/23.jpg" alt="Test" />
            <div>Prediction: {prediction}</div>
        </div>
    );
};

export default ISLTranslator;
