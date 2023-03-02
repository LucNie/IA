const brain = require('brain.js');
const net = new brain.NeuralNetwork({ hiddenLayers: [200] });
const fs = require('fs');
const { normalize } = require('path');
const path = require('path');

const pathMnist = path.join(__dirname, './data/mnist_test.csv');
const mnist = fs.readFileSync(pathMnist, 'utf8').split('\r').map(row => row.split(',').map(Number));

const config = {
    iterations: 250,
    // errorThresh: 0.005,
    log: true,
    logPeriod: 10,
    learningRate: 0.3,
    // momentum: 0.1,
    // callback: null,
    // callbackPeriod: 10,
    // timeout: Infinity
};

/** NOTE * Lucas 
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

// console.log("done normalizing", normalizemnist[0].input);

const normalizemnist2 = mnist.map(row => { // set > 127 to 1, < 127 to 0
    return {
        input: row.slice(1).map(x => x > 127 ? 1 : 0),
        output: [row[0] / 9]
    }
});

// console.log("done normalizing2", normalizemnist2[0].input);



// train the network
net.train(normalizemnist2, config);


// test the network
const test = mnist.slice(0, 100).map(row => { // test the first 100 rows
    return {
        input: row.slice(1).map(x => x > 127 ? 1 : 0),
        output: [row[0] / 9]
    }
});

for (let i = 0; i < 5; i++) {
    let _rand = Math.round(Math.random() * 100);
    const output = net.run(test[_rand].input);
    console.log("data : " + _rand," test output: ", output * 9, "expected: ", test[_rand].output * 9);
}

// save the network
const json = net.toJSON();
fs.writeFileSync('network.json', JSON.stringify(json));

// load the network
// const json = require('./network.json');
// const net = new brain.NeuralNetwork().fromJSON(json);

