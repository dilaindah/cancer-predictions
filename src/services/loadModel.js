const tf = require('@tensorflow/tfjs-node');

async function loadModel() {
    if (!process.env.MODEL_URL) {
        throw new Error('MODEL_URL is not defined in the .env file');
    }
    return tf.loadGraphModel(process.env.MODEL_URL);
}

module.exports = loadModel;
