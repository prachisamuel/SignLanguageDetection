import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';  // Required for Node.js

async function testTF() {
    try {
        console.log("Available Backends:", tf.engine().registryFactory);
        
        // Set the backend explicitly
        await tf.setBackend('tensorflow');
        
        console.log("Using Backend:", tf.getBackend());
        console.log("TensorFlow.js is ready!");
    } catch (error) {
        console.error("Error:", error);
    }
}

testTF();
