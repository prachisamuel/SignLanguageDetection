import React, { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import * as tf from "@tensorflow/tfjs";  // ✅ Import TensorFlow.js
import "@tensorflow/tfjs-backend-webgl"; // ✅ Import WebGL backend
import Webcam from "react-webcam";

const socket = io("http://localhost:5000"); // Connect to WebSocket server

const LABELS = [
    "Hello", "No", "Please", "Thankyou", "Yes", "ILoveYou"
];

const VideoCall = () => {
    const webcamRef = useRef(null);
    const [model, setModel] = useState(null);
    const [sign, setSign] = useState("");

    useEffect(() => {
        // loadModel();
        socket.on("offer", (offer) => handleOffer(offer));
        socket.on("answer", (answer) => handleAnswer(answer));
        socket.on("candidate", (candidate) => handleCandidate(candidate));

        const loadTF = async () => {
            await tf.ready(); // Ensure TensorFlow.js is fully initialized
    
            console.log("Available Backends Before:", tf.engine().backendNames());
    
            // Register WebGL first, then fallback to CPU
            try {
                await tf.setBackend("webgl");
            } catch (error) {
                console.warn("WebGL not available, switching to CPU.");
                await tf.setBackend("cpu");
            }
    
            console.log("Available Backends After:", tf.engine().backendNames());
            console.log("Using Backend:", tf.getBackend());

            window.tf = tf;
    
            loadModel();
        };
    
        loadTF();

        return () => {
            socket.off("offer");
            socket.off("answer");
            socket.off("candidate");
        };
    }, []);

    // Load the trained ML model
    const loadModel = async () => {
        try {
            await tf.setBackend("webgl"); // ✅ Set backend
            await tf.ready();             // ✅ Ensure TensorFlow.js is ready
            console.log("Using Backend:", tf.getBackend());

            const loadedModel = await tf.loadGraphModel(process.env.PUBLIC_URL + "/model/model.json");
            setModel(loadedModel);
            console.log("Model loaded successfully");
        } catch (error) {
            console.error("Error loading model:", error);
        }
    };

    // Process real-time video frames
    const detectHandGesture = async () => {
        if (model && webcamRef.current && webcamRef.current.video.readyState === 4) {
            const video = webcamRef.current.video;
    
            const prediction = await tf.tidy(async () => {
                let img = tf.browser.fromPixels(video);
                img = tf.image.resizeBilinear(img, [416, 416]);
                img = img.div(255.0);
                img = img.transpose([2, 0, 1]); // Convert NHWC -> NCHW
                img = img.expandDims(0);
    
                // ✅ Use executeAsync() instead of predict()
                return await model.executeAsync({ images: img });
            });
    
            const predictionData = await prediction.data();
            processPredictions(predictionData);
        }
    };    

    // Process model predictions
    const processPredictions = (predictions) => {
        const maxIndex = predictions.indexOf(Math.max(...predictions));
        const detectedSign = LABELS[maxIndex] || "Unknown"; // Replace LABELS with your class labels
        setSign(detectedSign);
    };
    

    // WebRTC Signaling Handlers (Placeholder Functions)
    const handleOffer = (offer) => {
        console.log("Received offer:", offer);
        // Implement WebRTC offer handling logic
    };

    const handleAnswer = (answer) => {
        console.log("Received answer:", answer);
        // Implement WebRTC answer handling logic
    };

    const handleCandidate = (candidate) => {
        console.log("Received ICE candidate:", candidate);
        // Implement ICE candidate handling logic
    };

    return (
        <div>
            <h1>Real-Time Sign Language Detection</h1>
            <Webcam ref={webcamRef} style={{ width: 640, height: 480 }} />
            <button onClick={detectHandGesture}>Detect Sign</button>
            <p>Detected Sign: {sign}</p>
        </div>
    );
};

export default VideoCall;
