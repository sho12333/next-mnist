const canvas = document.getElementById("myCanvas");
const ctx = canvas.getContext("2d");
const clearButton = document.getElementById("clear-button");
const resultCanvas = document.getElementById("resultCanvas");
const resultCtx = resultCanvas.getContext("2d");

let painting = false;

function startPosition(e) {
  painting = true;
  draw(e);
}

function finishedPosition() {
  painting = false;
  ctx.beginPath();
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  resultCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
}

function draw(e) {
  if (!painting) return;
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  ctx.strokeStyle = "black";

  ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
}

canvas.addEventListener("mousedown", startPosition);
canvas.addEventListener("mouseup", finishedPosition);
canvas.addEventListener("mousemove", draw);
clearButton.addEventListener("click", clearCanvas);

const sess = new onnx.InferenceSession();
const loadingModelPromise = sess.loadModel("./onnx_model.onnx");

async function updatePredictions() {
  const ctx = document.getElementById("myCanvas").getContext("2d");

  const imgData = ctx.getImageData(
    0,
    0,
    canvas.clientWidth,
    canvas.clientHeight
  );
  const input = new onnx.Tensor(new Float32Array(imgData.data), "float32");

  const outputMap = await sess.run([input]);
  const outputTensor = outputMap.values().next().value;
  const predictions = outputTensor.data;
  drawBarChart(predictions);
}

function drawBarChart(data) {
  resultCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);

  const barWidth = 25;
  const barSpacing = 15;
  const maxProbability = Math.max(...data);
  const canvasHeight = resultCanvas.height;
  const canvasWidth = resultCanvas.width;

  for (let i = 0; i < data.length; i++) {
    const barHeight = (data[i] * (canvasHeight - 50)) / maxProbability;
    const x = i * (barWidth + barSpacing) + barSpacing + 30;
    const y = canvasHeight - barHeight - 20;

    resultCtx.fillStyle = "blue";
    resultCtx.fillRect(x, y, barWidth, barHeight);

    resultCtx.fillStyle = "black";
    resultCtx.fillText(data[i].toFixed(2), x + barWidth / 2 - 10, y - 10);
  }

  resultCtx.beginPath();
  resultCtx.moveTo(10, 0);
  resultCtx.stroke();

  resultCtx.textAlign = "right";
  resultCtx.textBaseline = "middle";

  for (let i = 0; i <= 10; i++) {
    const y = canvasHeight - 20 - (i / 10) * (canvasHeight - 50);
    resultCtx.fillText((i / 10).toFixed(1), 25, y);
    resultCtx.beginPath();
    resultCtx.moveTo(30, y);
    resultCtx.lineTo(35, y);
    resultCtx.stroke();
  }

  resultCtx.beginPath();
  resultCtx.moveTo(30, canvasHeight - 20);
  resultCtx.stroke();

  resultCtx.textAlign = "center";
  resultCtx.textBaseline = "top";

  for (let i = 0; i < data.length; i++) {
    const x = i * (barWidth + barSpacing) + barSpacing + barWidth / 2;
    const y = canvasHeight - 15;
    resultCtx.fillText(`${i}`, x + 30, y);
  }
}
