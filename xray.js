const brain = require('brain.js');
const net = new brain.NeuralNetwork({ hiddenLayers: [20] });
const fs = require('fs');
const path = require('path');
const jimp = require('jimp');

process.env.NODE_OPTIONS = '--max-old-space-size=8192';

const config = {
    iterations: 1000,
    errorThresh: 0.0001,
    log: true,
    logPeriod: 10,
    learningRate: 0.3,
    // momentum: 0.1,
    // callback: null,
    // callbackPeriod: 10,
    // timeout: Infinity
};

const pathNORMAL = "/NORMAL"
const pathPNEUMONIA = "/PNEUMONIA"
const pathTEST = "/test"
const pathTRAIN = "/train"
const pathVAL = "/val"

const pathChestXray =path.join(__dirname, './chest_xray');
const pathChestXrayNormalized =path.join(__dirname, './chest_xray_normalized');

function normalize(aPathDir,aSubPathDir){
let file = fs.readdirSync(path.resolve(aPathDir + aSubPathDir));
// detele non jpeg file
file = file.filter(x => x.includes(".jpeg"));

for (let i = (file.length/2)-0.5; i < file.length; i++) {
    // si le pathdir cotient normal alors output = 0
    if (aSubPathDir.includes("NORMAL")) {
        var output = 0;
    } else {
        var output = 1;
    }
        // add directory path to each file name
        let pathFile = path.join(path.resolve(aPathDir + aSubPathDir), file[i]);
        // read the file
        
        image = fs.readFileSync(pathFile);
        image = jimp.decoders['image/jpeg'](image);
        image = new jimp(image);
        // resize the image 500x500 
        image.resize(500, 500);
        // grey scale the image
        image.greyscale();
        // convert to array
        image = image.bitmap.data;
        // normalize the image
        // image = image.map(x => x / 255);
        // create an object with input and output
        let data = {
            input: image,
            output: [output]
        }
        // remplace the .jpeg by .json
        file[i] = file[i].replace(".jpeg", "");

        // if the directory doesn't exist, create it
        if (!fs.existsSync(path.resolve(pathChestXrayNormalized + aSubPathDir))) {
            fs.mkdirSync(path.resolve(pathChestXrayNormalized + aSubPathDir));
        }

        // save the image to json with input and output
        fs.writeFileSync(path.resolve(pathChestXrayNormalized + aSubPathDir + "/" + file[i] + ".json"), JSON.stringify(data));
        console.log(i+1 + "/" + file.length + " " + path.resolve( aPathDir + aSubPathDir + file[i] + ".json"));
        // delete variable
        delete image;
        delete data;
        delete pathFile;
        delete output;
    }

}

// normalize(pathChestXray , path.join( pathTEST + pathNORMAL) ); //done
// normalize(pathChestXray , path.join( pathTEST + pathPNEUMONIA) );    //done
// normalize(pathChestXray , path.join( pathVAL + pathNORMAL) );    //done
// normalize(pathChestXray , path.join( pathVAL + pathPNEUMONIA) );   //done
// normalize(pathChestXray , path.join( pathTRAIN + pathNORMAL) );  //done
// normalize(pathChestXray , path.join( pathTRAIN + pathPNEUMONIA) );   //done
train()
function train(){

    const trainingData = [];
    const testData = [];
    const valData = [];

    // read all the file in the directory
    let fileTestNormal = fs.readdirSync(path.resolve(pathChestXrayNormalized + path.join( pathTEST + pathNORMAL)));
    let fileTestPneumonia = fs.readdirSync(path.resolve(pathChestXrayNormalized + path.join( pathTEST + pathPNEUMONIA)));
    let fileValNormal = fs.readdirSync(path.resolve(pathChestXrayNormalized + path.join( pathVAL + pathNORMAL)));
    let fileValPneumonia = fs.readdirSync(path.resolve(pathChestXrayNormalized + path.join( pathVAL + pathPNEUMONIA)));
    let fileTrainNormal = fs.readdirSync(path.resolve(pathChestXrayNormalized + path.join( pathTRAIN + pathNORMAL)));
    let fileTrainPneumonia = fs.readdirSync(path.resolve(pathChestXrayNormalized + path.join( pathTRAIN + pathPNEUMONIA)));

    // add the file to the array randomly
    for (let i = 0; i < fileTestNormal.length/10; i++) {
        let data = JSON.parse(fs.readFileSync(path.resolve(pathChestXrayNormalized + path.join( pathTEST + pathNORMAL + "/" + fileTestNormal[i]))));
        testData.push(data);
    }
    for (let i = 0; i < fileTestPneumonia.length/10; i++) {
        let data = JSON.parse(fs.readFileSync(path.resolve(pathChestXrayNormalized + path.join( pathTEST + pathPNEUMONIA + "/" + fileTestPneumonia[i]))));
        testData.push(data);
    }
        for (let i = 0; i < testData.length; i++) {
        testData[i].input = testData[i].input.data.map(x => x / 255);
    }
    console.log(testData[0].input)
    console.log("Test file extracted...")
    // add the file to the array test randomly
    // for (let i = 0; i < fileValNormal.length; i++) {
    //     let data = JSON.parse(fs.readFileSync(path.resolve(pathChestXrayNormalized + path.join( pathVAL + pathNORMAL + "/" + fileValNormal[i]))));
    //     valData.push(data);
    // }
    // for (let i = 0; i < fileValPneumonia.length; i++) {
    //     let data = JSON.parse(fs.readFileSync(path.resolve(pathChestXrayNormalized + path.join( pathVAL + pathPNEUMONIA + "/" + fileValPneumonia[i]))));
    //     valData.push(data);
    // }
    // add the file to the array train randomly
    for (let i = 0; i < fileTrainNormal.length/10; i++) {
        let data = JSON.parse(fs.readFileSync(path.resolve(pathChestXrayNormalized + path.join( pathTRAIN + pathNORMAL + "/" + fileTrainNormal[i]))));
        trainingData.push(data);
    }
    for (let i = 0; i < fileTrainPneumonia.length/10; i++) {
        let data = JSON.parse(fs.readFileSync(path.resolve(pathChestXrayNormalized + path.join( pathTRAIN + pathPNEUMONIA + "/" + fileTrainPneumonia[i]))));
        trainingData.push(data);
    }
    for (let i = 0; i < trainingData.length; i++) {
        trainingData[i].input = trainingData[i].input.data.map(x => x / 255); 
    }
    // for (let i = 0; i < testData.length; i++) { //Fisher-Yates shuffle
    //     let j = Math.floor(Math.random() * i);
    //     let temp = testData[i];
    //     testData[i] = testData[j];
    //     testData[j] = temp;
    // }
    console.log("Train file extracted...")


    // // for (let i = 0; i < trainingData.length; i++) { //Fisher-Yates shuffle
    // //     let j = Math.floor(Math.random() * i);
    // //     let temp = trainingData[i];
    // //     trainingData[i] = trainingData[j];
    // //     trainingData[j] = temp;
    // // }
    // console.log("data randomized")
    // // normalize the data


    console.log("data normalized")
    
    // create the neural network
    const net = new brain.NeuralNetwork({hiddenLayers : [25]});

    // train the neural network
    net.train(trainingData, config);

    // test the neural network
    let correct = 0;
    let wrong = 0;
    for (let i = 0; i < testData.length; i++) {
        let output = net.run(testData[i].input);
        let result = output[0] > output[1] ? 0 : 1;
        if (result == testData[i].output[0]) {
            correct++;
        } else {
            wrong++;
        }
    }
    console.log("correct: " + correct);
    console.log("wrong: " + wrong);
    console.log("accuracy: " + (correct / (correct + wrong)) * 100 + "%");

    // // save the neural network
    // const json = net.toJSON();
    // fs.writeFileSync(path.resolve(pathChestXrayNormalized + "model.json"), JSON.stringify(json));
}