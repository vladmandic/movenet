const fs = require('fs');
const path = require('path');
const process = require('process');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');
const canvas = require('canvas');

const modelOptions = {
  modelPath: 'file://model-multipose/movenet-multipose.json', // https://storage.googleapis.com/movenet/MoveNet.MultiPose%20Model%20Card.pdf
  minConfidence: 0.2,
};

const bodyParts = ['nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'];

// save image with processed results
async function saveImage(res, img) {
  // create canvas
  const c = new canvas.Canvas(img.inputShape[1], img.inputShape[0]);
  const ctx = c.getContext('2d');

  // load and draw original image
  const original = await canvas.loadImage(img.fileName);
  ctx.drawImage(original, 0, 0, c.width, c.height);
  // const fontSize = Math.trunc(c.width / 50);
  const fontSize = Math.round((c.width * c.height) ** (1 / 2) / 80);
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'white';
  ctx.font = `${fontSize}px "Segoe UI"`;

  // draw all detected objects
  for (const person of res) {
    for (const obj of person.parts) {
      ctx.fillStyle = 'black';
      ctx.fillText(`${Math.round(100 * obj.score)}% ${obj.label}`, obj.x + 1, obj.y + 1);
      ctx.fillStyle = 'white';
      ctx.fillText(`${Math.round(100 * obj.score)}% ${obj.label}`, obj.x, obj.y);
    }
    ctx.stroke();

    const connectParts = (parts, color) => {
      ctx.strokeStyle = color;
      ctx.beginPath();
      for (let i = 0; i < parts.length; i++) {
        const part = person.parts.find((a) => a.label === parts[i]);
        if (part) {
          if (i === 0) ctx.moveTo(part.x, part.y);
          else ctx.lineTo(part.x, part.y);
        }
      }
      ctx.stroke();
    };

    connectParts(['nose', 'leftEye', 'rightEye', 'nose'], '#99FFFF');
    connectParts(['rightShoulder', 'rightElbow', 'rightWrist'], '#99CCFF');
    connectParts(['leftShoulder', 'leftElbow', 'leftWrist'], '#99CCFF');
    connectParts(['rightHip', 'rightKnee', 'rightAnkle'], '#9999FF');
    connectParts(['leftHip', 'leftKnee', 'leftAnkle'], '#9999FF');
    connectParts(['rightShoulder', 'leftShoulder', 'leftHip', 'rightHip', 'rightShoulder'], '#9900FF');
  }

  // write canvas to jpeg
  const outImage = `outputs/${path.basename(img.fileName)}`;
  const out = fs.createWriteStream(outImage);
  out.on('finish', () => log.state('Created output image:', outImage, 'size:', [c.width, c.height]));
  out.on('error', (err) => log.error('Error creating image:', outImage, err));
  const stream = c.createJPEGStream({ quality: 0.6, progressive: true, chromaSubsampling: true });
  stream.pipe(out);
}

// load image from file and prepares image tensor that fits the model
async function loadImage(fileName, inputSize) {
  const data = fs.readFileSync(fileName);
  const obj = tf.tidy(() => {
    const buffer = tf.node.decodeImage(data);
    const expand = buffer.expandDims(0);
    // @ts-ignore
    const resize = tf.image.resizeBilinear(expand, [inputSize, inputSize]);
    const cast = tf.cast(resize, 'int32');
    const tensor = cast;
    const img = { fileName, tensor, inputShape: buffer?.shape, modelShape: tensor?.shape, size: buffer?.size };
    return img;
  });
  return obj;
}

async function processResults(res, img) {
  const data = res.arraySync();
  log.info('Tensor output', res.shape);
  // log.data(data);
  res.dispose();
  const people = [];
  for (let p = 0; p < data[0].length; p++) {
    const kpt = data[0][p];
    const score = kpt[51 + 4];
    // eslint-disable-next-line no-continue
    if (score < modelOptions.minConfidence) continue;
    const parts = [];
    for (let i = 0; i < 17; i++) {
      const part = {
        id: i,
        label: bodyParts[i],
        score: kpt[3 * i + 2],
        xRaw: kpt[3 * i + 1],
        yRaw: kpt[3 * i + 0],
        x: Math.trunc(kpt[3 * i + 1] * img.inputShape[1]),
        y: Math.trunc(kpt[3 * i + 0] * img.inputShape[0]),
      };
      parts.push(part);
    }
    const boxRaw = [kpt[51 + 1], kpt[51 + 0], kpt[51 + 3] - kpt[51 + 1], kpt[51 + 2] - kpt[51 + 0]];
    people.push({
      id: p,
      score,
      boxRaw,
      box: boxRaw.map((a) => Math.trunc(a * img.inputShape[1])),
      parts,
    });
  }
  return people;
}

async function main() {
  log.header();

  // init tensorflow
  await tf.enableProdMode();
  await tf.setBackend('tensorflow');
  await tf.ENV.set('DEBUG', false);
  await tf.ready();

  // load model
  const model = await tf.loadGraphModel(modelOptions.modelPath);
  log.info('Loaded model', modelOptions, 'tensors:', tf.engine().memory().numTensors, 'bytes:', tf.engine().memory().numBytes);
  // @ts-ignore
  log.info('Model Signature', model.signature);

  // load image and get approprite tensor for it
  let inputSize = Object.values(model.modelSignature['inputs'])[0].tensorShape.dim[2].size;
  if (inputSize === -1) inputSize = 256;
  const imageFile = process.argv.length > 2 ? process.argv[2] : null;
  if (!imageFile || !fs.existsSync(imageFile)) {
    log.error('Specify a valid image file');
    process.exit();
  }
  const img = await loadImage(imageFile, inputSize);
  log.info('Loaded image:', img.fileName, 'inputShape:', img.inputShape, 'modelShape:', img.modelShape, 'decoded size:', img.size);

  // run actual prediction
  const t0 = process.hrtime.bigint();
  // for (let i = 0; i < 99; i++) model.execute(img.tensor); // benchmarking
  const res = model.execute(img.tensor);
  const t1 = process.hrtime.bigint();
  log.info('Inference time:', Math.round(parseInt((t1 - t0).toString()) / 1000 / 1000), 'ms');

  // process results
  const results = await processResults(res, img);
  const t2 = process.hrtime.bigint();
  log.info('Processing time:', Math.round(parseInt((t2 - t1).toString()) / 1000 / 1000), 'ms');

  // print results
  log.data('Results:', results);

  // save processed image
  await saveImage(results, img);
}

main();
