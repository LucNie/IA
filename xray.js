const fs = require('fs');
const path = require('path');
const Jimp = require('jimp');
const brain = require('brain.js');
// $env:NODE_OPTIONS="--max-old-space-size=16384"
// use dotenv to load environment variables from a .env file
require('dotenv').config();
const inputFolder = path.join(__dirname, 'chest-Xray');
const outputFolder = path.join(__dirname, 'normalized-chest-Xray');
// $env:NODE_OPTIONS="--max-old-space-size=16392"

const net = new brain.NeuralNetwork({
  // Nombre de couches cachées et nombre de neurones dans chaque couche
  hiddenLayers: [32, 32, 32],

});

if (process.env.IMAGE_NORMALIZE === "false" ){
  trainingIA();
  evaluate(net);
}


async function normalizeImagesInFolder(inputFolder, outputFolder) {
  const files = fs.readdirSync(inputFolder);


  // Créer le dossier de sortie s'il n'existe pas
  if (!fs.existsSync(outputFolder)) {
    fs.mkdirSync(outputFolder);
  }

  for (const file of files) {

    // Ignorer les fichiers qui ne sont pas des images
    if (!file.match(/\.(jpg|jpeg|png)$/)) {
      continue;
    }

    const inputFile = path.join(inputFolder, file);
    const stats = fs.statSync(inputFile);

    if (!stats.isFile()) {
      continue;
    }

    const normalizedImage = await normalizeImage(inputFile);
    const outputFile = path.join(outputFolder, file);

    await normalizedImage.writeAsync(outputFile);
  }
  console.log(`Images normalisées dans le dossier ${outputFolder}`);
}

async function normalizeImage(imagePath) {
  const image = await Jimp.read(imagePath);

  // Convertir l'image en niveaux de gris

  // Appliquer un filtre de contraste automatique

  // redimensionner l'image
  image.resize(1000, 1000);
  image.greyscale();
  // to jpeg
  image.getBufferAsync(Jimp.MIME_JPEG);

  return image;
}
if (process.env.IMAGE_NORMALIZE === "true"){

// create dir if not exist
if (!fs.existsSync(outputFolder)) {
  fs.mkdirSync(outputFolder);
  fs.mkdirSync(outputFolder + '/test');
  fs.mkdirSync(outputFolder + '/test/NORMAL');
  fs.mkdirSync(outputFolder + '/test/PNEUMONIA');
  fs.mkdirSync(outputFolder + '/train');
  fs.mkdirSync(outputFolder + '/train/NORMAL');
  fs.mkdirSync(outputFolder + '/train/PNEUMONIA');
  fs.mkdirSync(outputFolder + '/val');
  fs.mkdirSync(outputFolder + '/val/NORMAL');
  fs.mkdirSync(outputFolder + '/val/PNEUMONIA');
}
normalizeImagesInFolder(path.resolve(inputFolder + '/test/NORMAL'), path.resolve(outputFolder + '/test/NORMAL')); //done
normalizeImagesInFolder(path.resolve(inputFolder + '/test/PNEUMONIA'), path.resolve(outputFolder + '/test/PNEUMONIA')); //done
normalizeImagesInFolder(path.resolve(inputFolder + '/train/NORMAL'), path.resolve(outputFolder + '/train/NORMAL'));  //done
normalizeImagesInFolder(path.resolve(inputFolder + '/train/PNEUMONIA'), path.resolve(outputFolder + '/train/PNEUMONIA'));  //done
normalizeImagesInFolder(path.resolve(inputFolder + '/val/NORMAL'), path.resolve(outputFolder + '/val/NORMAL'));  //done
normalizeImagesInFolder(path.resolve(inputFolder + '/val/PNEUMONIA'), path.resolve(outputFolder + '/val/PNEUMONIA'));  //done
}
console.log('Normalisation des images en cours...');
function trainingIA() {

    // Charger les données d'entraînement et de validation
    const dataset = loadData(path.join(__dirname, 'normalized-chest-Xray/train/NORMAL'), path.join(__dirname, 'normalized-chest-Xray/train/PNEUMONIA'));


    // Entraîner le réseau de neurones
    net.train(dataset, {
      errorThresh: 0.05,
      iterations: 200,
      log: true,
      logPeriod: 1,
      learningRate: 0.3,
    });

    // Évaluer le réseau de neurones sur l'ensemble de validation
    const accuracy = brain.util.getBinaryAccuracy(net, validationData);
    console.log(`Accuracy: ${accuracy}`);

    // Sauvegarder le réseau de neurones
    const json = net.toJSON();
    fs.writeFileSync('xray.json', JSON.stringify(json));

}

// Évaluer le réseau de neurones sur un ensemble de données
function evaluate(net) {

  const testData = loadData(path.join(__dirname, 'normalized-chest-Xray/test/NORMAL'), path.join(__dirname, 'normalized-chest-Xray/test/PNEUMONIA'));

  let correct = 0;
  let wrong = 0;

  for (let i = 0; i < testData.length; i++) {
    const output = net.run(testData[i].input);
    console.log(`Test ${i} - Output: ${output} - Expected: ${testData[i].output}`);
    if (output === Math.round(testData[i].output)) {
      correct++;
    }
    else {
      wrong++;
    }
  }

  console.log(`Correct: ${correct} - Wrong: ${wrong}` + ` - Accuracy: ${correct / (correct + wrong)}`);

}

function loadData(healthyFolder,pneumoniaFolder,ignoreDiviser){
  const dataset = [];

  const healthyFiles = fs.readdirSync(healthyFolder);
  const pneumoniaFiles = fs.readdirSync(pneumoniaFolder);
  console.log(healthyFiles.length);
  console.log(pneumoniaFiles.length);
  console.log(Math.floor(healthyFiles.length / process.env.DATA_DIVISER));
  console.log(Math.floor(pneumoniaFiles.length / process.env.DATA_DIVISER));

  NUMBEROFELEMENTS = healthyFiles.length;
  if (!ignoreDiviser) {
    NUMBEROFELEMENTS = Math.floor(healthyFiles.length / process.env.DATA_DIVISER);
  }

  console.log("number of elements: " + NUMBEROFELEMENTS);  

  for (let i = 0; i < NUMBEROFELEMENTS; i++) {
    const file = healthyFiles[i];
    const filePath = path.join(healthyFolder, file);
    const buffer = fs.readFileSync(filePath);
    const rawimage = Jimp.decoders['image/jpeg'](buffer);
    const image = new Jimp(rawimage).greyscale().resize(Number(process.env.IMAGE_HEIGHT), Number(process.env.IMAGE_WIDTH));

    // to json parse
    const jsonbuffer = Array.prototype.slice.call(image.bitmap.data).map((value) => value / 255); // 0-255 to 0-1
    const jsonNormal = [];
    // use 1 data of 4 (4 = 1 pixel)
    for (let i = 0; i < jsonbuffer.length; i++) {
      const pixel = jsonbuffer.slice(i, i + 4);
      jsonNormal.push(pixel[0]);
    }

    dataset.push({
      input: jsonNormal,
      output: [0], // "poumon sain"
    });

    console.log("poumon sain : " + i + " / " + NUMBEROFELEMENTS);
  }

  for (let i = 0; i < NUMBEROFELEMENTS; i++) {
    const file = pneumoniaFiles[i];
    const filePath = path.join(pneumoniaFolder, file);
    const buffer = fs.readFileSync(filePath);
    const rawimage = Jimp.decoders['image/jpeg'](buffer);
    const image = new Jimp(rawimage).greyscale().resize(Number(process.env.IMAGE_HEIGHT), Number(process.env.IMAGE_WIDTH));
    // to json parse
    const jsonbuffer = Array.prototype.slice.call(image.bitmap.data).map((value) => value / 255); // 0-255 to 0-1
    const jsonPneumonia = [];
    // use 1 data of 4 (4 = 1 pixel) to keep only the greyscale and ignore alpha/color 
    for (let i = 0; i < jsonbuffer.length; i++) {
      const pixel = jsonbuffer.slice(i, i + 4);
      jsonPneumonia.push(pixel[0]);
    }

    dataset.push({
      input: jsonPneumonia,
      output: [1], // "pneumonie"
    });


    
    console.log("poumon atteint de pneumonie" + i + " / " + NUMBEROFELEMENTS);
  }

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


