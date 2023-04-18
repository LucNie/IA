const fs = require('fs');
const path = require('path');
const Jimp = require('jimp');
const brain = require('brain.js');
const flatted = require('flatted');
// $env:NODE_OPTIONS="--max-old-space-size=16384"
// use dotenv to load environment variables from a .env file
require('dotenv').config();
const inputFolder = path.join(__dirname, 'chest-Xray');
const outputFolder = path.join(__dirname, 'normalized-chest-Xray');
// $env:NODE_OPTIONS="--max-old-space-size=16392"

const net = new brain.NeuralNetworkGPU({

hiddenLayers: [2048, 1024, 512]
// layers: [
//   { type: 'input', size: 16384 },
//   { type: 'dense', size: 128, activation: 'sigmoid' },
//   { type: 'dense', size: 64, activation: 'sigmoid' },
//   { type: 'dense', size: 5, activation: 'sigmoid' }
// ],
});

if (process.env.IMAGE_VIEW === "true") {
  viewImage();
  // wait 5 seconds*
  console.log('Création des images de visualisation en cours...');
  setTimeout(function () {
    if (process.env.IMAGE_NORMALIZE === "false") {
      trainingIA();
    }
  }, 10000);

} else {
  if (process.env.IMAGE_NORMALIZE === "false") {
    trainingIA();
  }
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

    if (!stats.isFile()) { // Ignorer les dossiers
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

  // redimensionner l'image
  image.resize(1000, 1000);
  image.greyscale();
  // to jpeg
  image.getBufferAsync(Jimp.MIME_JPEG);

  return image;
}
if (process.env.IMAGE_NORMALIZE === "true") {

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
    fs.mkdirSync(outputFolder + '/view');
    fs.mkdirSync(outputFolder + '/view/NORMAL');
    fs.mkdirSync(outputFolder + '/view/PNEUMONIA');
  }
  normalizeImagesInFolder(path.resolve(inputFolder + '/test/NORMAL'), path.resolve(outputFolder + '/test/NORMAL')); //done
  normalizeImagesInFolder(path.resolve(inputFolder + '/test/PNEUMONIA'), path.resolve(outputFolder + '/test/PNEUMONIA')); //done
  normalizeImagesInFolder(path.resolve(inputFolder + '/train/NORMAL'), path.resolve(outputFolder + '/train/NORMAL'));  //done
  normalizeImagesInFolder(path.resolve(inputFolder + '/train/PNEUMONIA'), path.resolve(outputFolder + '/train/PNEUMONIA'));  //done
  normalizeImagesInFolder(path.resolve(inputFolder + '/val/NORMAL'), path.resolve(outputFolder + '/val/NORMAL'));  //done
  normalizeImagesInFolder(path.resolve(inputFolder + '/val/PNEUMONIA'), path.resolve(outputFolder + '/val/PNEUMONIA'));  //done
}
console.log('Normalisation des images en cours...');

async function viewImage() {
  console.log('Création des images de visualisation en cours...');
  const filesNormals = fs.readdirSync(inputFolder + '/test/NORMAL')
  const filesPneumonia = fs.readdirSync(inputFolder + '/test/PNEUMONIA')

  console.log("filesNormals", filesNormals);
  console.log("filesPneumonia", filesPneumonia);

  // take the first image
  const fileNormal = filesNormals[Math.floor(Math.random() * filesNormals.length)];
  const filePneumonia = filesPneumonia[Math.floor(Math.random() * filesPneumonia.length)];
  const inputFileNormal = path.join(inputFolder + '/test/NORMAL', fileNormal);
  const inputFilePneumonia = path.join(inputFolder + '/test/PNEUMONIA', filePneumonia);

  console.log("inputFileNormal", inputFileNormal);
  console.log("inputFilePneumonia", inputFilePneumonia);

  const imageNormal = await Jimp.read(inputFileNormal)
  const imagePneumonia = await Jimp.read(inputFilePneumonia)
  console.log("modification des images en cours...")

  imageNormal.greyscale()
  imageNormal.resize(Number(process.env.IMAGE_HEIGHT), Number(process.env.IMAGE_WIDTH))
  imageNormal.threshold({ max: 150 })
  imageNormal.blur(1);

  imagePneumonia.greyscale()
  imagePneumonia.resize(Number(process.env.IMAGE_HEIGHT), Number(process.env.IMAGE_WIDTH))
  imagePneumonia.threshold({ max: 150 })
  imagePneumonia.blur(1);



  // get jpeg buffer
  const imageNormalBuffer = await imageNormal.getBufferAsync(Jimp.MIME_JPEG);
  const imagePneumoniaBuffer = await imagePneumonia.getBufferAsync(Jimp.MIME_JPEG);
  
  await fs.promises.writeFile(path.join( outputFolder + '/view/NORMAL', 'viewNormal.jpg'), imageNormalBuffer);
  await fs.promises.writeFile(path.join( outputFolder + '/view/PNEUMONIA', 'viewPneumonia.jpg'), imagePneumoniaBuffer);

  console.log("Size of imageNormalBuffer", imageNormalBuffer.length);

  console.log('Images de visualisation enregistrées dans le dossier view');
}


function trainingIA() {

  // Charger les données d'entraînement et de validation

  IAStartLog()



  // Entraîner le réseau de neurones

  for (let i = 0; i < process.env.ROTATION_NUMBER; i++) {

    console.log("Start rotation number ", i);
    const dataset = loadData(path.join(__dirname, 'normalized-chest-Xray/train/NORMAL'), path.join(__dirname, 'normalized-chest-Xray/train/PNEUMONIA'));
  net.train(dataset, {
    errorThreshold: Number(process.env.IA_ERROR_THRESHOLD),
    iterations: Number(process.env.IA_ITERATIONS),
    log: (process.env.IA_LOG === 'true'),
    logPeriod: Number(process.env.IA_LOG_PERIOD),
    learningRate: Number(process.env.IA_LEARNING_RATE),
    momentum: Number(process.env.IA_MOMENTUM),
    // activation: Number(process.env.IA_ACTIVATION),
  });

  console.log("End rotation number ", i);
}

  // Évaluer le réseau de neurones sur l'ensemble de validation
  // const accuracy = brain.util.getBinaryAccuracy(net, validationData);
  // console.log(`Accuracy: ${accuracy}`);

  // Sauvegarder le réseau de neurones
  const json = net.toJSON();


  let out="[";
  for(let indx=0;indx<json.length-1;indx++){
    out+=JSON.stringify(json[indx],null,4)+",";
  }
  out+=JSON.stringify(json[json.length-1],null,4)+"]";

  fs.writeFileSync("Xray_T" + process.env.IMAGE_HEIGHT+'_LR'+ process.env.IA_LEARNING_RATE+'_MO'+process.env.IA_MOMENTUM+'_ERR'+ process.env.IA_ERROR_THRESHOLD + '.json', out);
  console.log("IA saved");
 
  evaluate(net);
}

// Évaluer le réseau de neurones sur un ensemble de données
function evaluate(net) {

  const testData = loadData(path.join(__dirname, 'normalized-chest-Xray/test/NORMAL'), path.join(__dirname, 'normalized-chest-Xray/test/PNEUMONIA'));


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

  const valData = loadData(path.join(__dirname, 'normalized-chest-Xray/val/NORMAL'), path.join(__dirname, 'normalized-chest-Xray/val/PNEUMONIA'));

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


  console.log(`Correct: ${correct} - Wrong: ${incorrect}` + ` - Accuracy: ${correct / (correct + incorrect)}`);

}

function loadData(healthyFolder, pneumoniaFolder, ignoreDiviser) {
  const dataset = [];

  const healthyFiles = fs.readdirSync(healthyFolder);
  const pneumoniaFiles = fs.readdirSync(pneumoniaFolder);


  NUMBEROFELEMENTS = healthyFiles.length;
  if (!ignoreDiviser) {
    NUMBEROFELEMENTS = Math.floor(healthyFiles.length / (process.env.DATA_DIVISER * process.env.ROTATION_NUMBER)) ;
  }

  console.log("number of elements: " + NUMBEROFELEMENTS);
  console.log("Start loading healthy data");

  for (let i = 0; i < NUMBEROFELEMENTS; i++) {
    const file = healthyFiles[Math.floor(Math.random() * healthyFiles.length)];
    const filePath = path.join(healthyFolder, file);
    const buffer = fs.readFileSync(filePath);
    const rawimage = Jimp.decoders['image/jpeg'](buffer);
    const image = new Jimp(rawimage).greyscale().resize(Number(process.env.IMAGE_HEIGHT), Number(process.env.IMAGE_WIDTH)).threshold({ max: 150 }).blur(1);

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


    // console.log("poumon sain : " + i + " / " + NUMBEROFELEMENTS);
  }

  console.log("loading healthy data done")
  console.log("Start loading pneumonia data");

  for (let i = 0; i < NUMBEROFELEMENTS; i++) {
    const file = pneumoniaFiles[i];
    const filePath = path.join(pneumoniaFolder, file);
    const buffer = fs.readFileSync(filePath);
    const rawimage = Jimp.decoders['image/jpeg'](buffer);
    const image = new Jimp(rawimage).greyscale().resize(Number(process.env.IMAGE_HEIGHT), Number(process.env.IMAGE_WIDTH)).threshold({max: 150}).blur(1);
    
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
  console.log("loading pneumonia data done")

  console.log("dataset loaded !");

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

function IAStartLog() {
  console.log(`Image size: \x1b[33m%s\x1b[0m`, ` ${process.env.IMAGE_HEIGHT}x${process.env.IMAGE_WIDTH}`);
  console.log(`Image normalization: \x1b[33m%s\x1b[0m`, `${process.env.IMAGE_NORMALIZE}`);
  console.log(`IA error threshold: \x1b[33m%s\x1b[0m`, `${process.env.IA_ERROR_THRESHOLD}`);
  console.log(`IA iterations: \x1b[33m%s\x1b[0m`, `${process.env.IA_ITERATIONS}`);
  console.log(`IA learning rate: \x1b[33m%s\x1b[0m`, `${process.env.IA_LEARNING_RATE}`);
  console.log(`IA momentum: \x1b[33m%s\x1b[0m`, `${process.env.IA_MOMENTUM}`);
  console.log(`IA activation function: \x1b[33m%s\x1b[0m`, `${process.env.IA_ACTIVATION}`);
  console.log(`IA number of hidden layers: \x1b[33m%s\x1b[0m`, `${process.env.IA_HIDDEN_LAYER}`);
  console.log(`IA logging: \x1b[33m%s\x1b[0m`, `${process.env.IA_LOG}`);
  console.log(`IA logging period: \x1b[33m%s\x1b[0m`, `${process.env.IA_LOG_PERIOD}`);
}

