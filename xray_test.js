const fs = require('fs');
const path = require('path');
const Jimp = require('jimp');
const brain = require('brain.js');

require('dotenv').config();
const inputFolder = path.join(__dirname, 'chest-Xray');
const outputFolder = path.join(__dirname, 'normalized-chest-Xray');

const net = new brain.NeuralNetwork();
const networkName = "Xray_T128_LR0.1_MO0.1_ERR0.1.json"
const network = JSON.parse(fs.readFileSync(path.join(__dirname, networkName), 'utf8'));
const testData = loadData(path.join(__dirname, 'normalized-chest-Xray/test/NORMAL'), path.join(__dirname, 'normalized-chest-Xray/test/PNEUMONIA'),false);
const valData = loadData(path.join(__dirname, 'normalized-chest-Xray/val/NORMAL'), path.join(__dirname, 'normalized-chest-Xray/val/PNEUMONIA'),false);


test();
function test(){

    
    net.fromJSON(network);

    let correct = 0;
    let incorrect = 0;

    for (let i = 0; i < testData.length; i++) {
        const test = testData[i];
        const result = net.run(test.input);
        const output = result[0] > 0.5 ? 1 : 0;
        const expected = test.output[0];
        if (output === expected) {
            correct++;
        } else {
            incorrect++;
        }
    }
    console.log("test data");
    console.log("correct: " + correct);
    console.log("incorrect: " + incorrect);
    console.log("accuracy: " + (correct / (correct + incorrect)) * 100 + "%");

    // validation
    correct = 0;
    incorrect = 0;

    for (let i = 0; i < valData.length; i++) {
        const test = valData[i];
        const result = net.run(test.input);
        const output = result[0] > 0.5 ? 1 : 0;
        const expected = test.output[0];
        if (output === expected) {
            correct++;
        } else {
            incorrect++;
        }
    }

    console.log("validation data");
    console.log("correct: " + correct);
    console.log("incorrect: " + incorrect);
    console.log("accuracy: " + (correct / (correct + incorrect)) * 100 + "%");


}

function loadData(healthyFolder, pneumoniaFolder, ignoreDiviser) {
    const dataset = [];
  
    const healthyFiles = fs.readdirSync(healthyFolder);
    const pneumoniaFiles = fs.readdirSync(pneumoniaFolder);
    // decompose network name to get the number of elements T:
    const tailleImage = Number(networkName.split("_")[1].split("T")[1]);
    console.log("taille image: " + tailleImage);



    NUMBEROFELEMENTS = healthyFiles.length;
    if (!ignoreDiviser) {
      NUMBEROFELEMENTS = Math.floor(healthyFiles.length ) ;
    }
  
    console.log("number of elements: " + NUMBEROFELEMENTS);
    console.log("loading healthy files");
  
    for (let i = 0; i < NUMBEROFELEMENTS; i++) {
      const file = healthyFiles[i];
      const filePath = path.join(healthyFolder, file);
      const buffer = fs.readFileSync(filePath);
      const rawimage = Jimp.decoders['image/jpeg'](buffer);
      const image = new Jimp(rawimage).greyscale().resize(tailleImage, tailleImage).threshold({ max: 150 }).blur(1);
  
      // write first image to file view /normalizes-chest-Xray/view/normal/0.jpg
  
      // to json parse
      const jsonbuffer = Array.prototype.slice.call(image.bitmap.data).map((value) => value / 255); // 0-255 to 0-1
      const jsonNormal = [];
      // use 1 data of 4 (4 = 1 pixel)
      for (let i = 0; i < jsonbuffer.length; i += 4) {
        jsonNormal.push(jsonbuffer[i]);
      }
  
      dataset.push({
        input: jsonNormal,
        output: [0], // "poumon sain"
      });
  
  
  
    }

    console.log("healthy files loaded");
    console.log("loading pneumonia files");
  
    for (let i = 0; i < NUMBEROFELEMENTS; i++) {
      const file = pneumoniaFiles[i];
      const filePath = path.join(pneumoniaFolder, file);
      const buffer = fs.readFileSync(filePath);
      const rawimage = Jimp.decoders['image/jpeg'](buffer);
      const image = new Jimp(rawimage).greyscale().resize(tailleImage, tailleImage).threshold({max: 150}).blur(1);
      
      // to json parse
      const jsonbuffer = Array.prototype.slice.call(image.bitmap.data).map((value) => value / 255); // 0-255 to 0-1
      const jsonPneumonia = [];
  
      // use 1 data of 4 (4 = 1 pixel) to keep only the greyscale and ignore alpha/color 
      for (let i = 0; i < jsonbuffer.length; i += 4) {
        jsonPneumonia.push(jsonbuffer[i]);
      }    
  
      dataset.push({
        input: jsonPneumonia,
        output: [1], // "pneumonie"
      });
  
    }
  
    console.log("pneumonia files loaded");
    console.log("dataset loaded");
  
    return shuttlleArray(dataset);
  }
  

  function shuttlleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  
    console.log("dataset shuffled");
  
    return array;
  
  }