const brain = require('brain.js');
const net = new brain.NeuralNetwork({});
const fs = require('fs');
const path = require('path');

const pathMnist = path.join(__dirname, './data/mnist_test.csv');
const mnist = fs.readFileSync(pathMnist, 'utf8').split('\r').map(row => row.split(',').map(Number));


/** NOTE
 * mnist each row is a 28x28 image, and the last column is the label
 * the first column is the label, and the rest are the pixels 
 */

console.log('length of mnist: ', mnist.length);
console.log('length of mnist[0]: ', mnist[0].length);
for (let i = 0; i < mnist.length; i++) { // verify is all is a number
    for (let j = 0; j < mnist[i].length; j++) {
        // result is a number ?
        // console.log('mnist[' + i + '][' + j + '] is a number: ', typeof mnist[i][j] === 'number');
        if (typeof mnist[i][j] !== 'number') {
            console.log('mnist[' + i + '][' + j + '] is not a number');
        }
    }
}

// process.exit();

// train the network
const trainingData = mnist.map(row => {
    const input = row.slice(1);
    const output = row[0];
    return {
        input,
        output
    }
});

// console.log('trainingData[0]: ', trainingData[52].output);

net.train(trainingData, {  
    iterations: 100, // the maximum times to iterate the training data --> number greater than 0
    // errorThresh: 0.005, // error threshold to reach
    log: true, // true to use console.log, when a function is supplied it is used --> Either true or a function
    logPeriod: 10,
    learningRate: 0.3,
    // momentum: 0.1, // momentum, multiplier of last delta change (max: 1)
    // callback: null,
    // callbackPeriod: 10,
    // timeout: Infinity
});


// test the network
const test = mnist.slice(0, 100).map(row => {
    const input = row.slice(1); // the first column is the label
    const output = row[0]; // the rest are the pixels
    return {
        input,
        output
    }
});

// console.log(test[0].input);

const result = net.run(test[0].input);
console.log(result);

// save the network
// const pathNet = path.join(__dirname, './data/net.json');    
// const json = net.toJSON();