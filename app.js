const brain = require('brain.js');
const net = new brain.NeuralNetwork( { hiddenLayers: [300] } );
const fs = require('fs');
const { normalize } = require('path');
const path = require('path');

const pathMnist = path.join(__dirname, './data/mnist_test.csv');
const mnist = fs.readFileSync(pathMnist, 'utf8').split('\r').map(row => row.split(',').map(Number));

/**
 * mnist is a 28x28 image of a number from 0-9
 * the first column is the number the image represents
 * the rest of the columns are the pixel values
 */

const normalizemnist = mnist.map(row => {
    return {
        input: row.slice(1) / 255,
        output: [row[0] / 9] 
    }
});

const normalizemnist2 = mnist.map(row => { // set > 127 to 1, < 127 to 0
    return {
        input: row.slice(1).map(x => x > 127 ? 1 : 0),
        output: [row[0] / 9]
    }
});


const config = {
    iterations: 100,
    // errorThresh: 0.005,
    log: true,
    // logPeriod: 10,
    learningRate: 0.3,
    // momentum: 0.1,
    // callback: null,
    // callbackPeriod: 10,
    // timeout: Infinity
};
// train with gpu
net.train(normalizemnist2, config);


// test the network
const test = mnist.slice(0, 100).map(row => { // test the first 100 rows
    return {
        input: row.slice(1) / 255,
        output: [row[0] / 9]
    }
});


const output = net.run(test[99].input);
console.log("test output: ", output * 9 , "expected: ", test[99].output * 9);