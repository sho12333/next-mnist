// ページの読み込みが完了したらコールバック関数が呼ばれる
// ※コールバック: 第2引数の無名関数(=関数名が省略された関数)
window.addEventListener('load', () => {
    const canvas = document.querySelector('#draw-area');
    //console.log(canvas.width)
    // contextを使ってcanvasに絵を書いていく
    const context = canvas.getContext('2d');
  
    // 直前のマウスのcanvas上のx座標とy座標を記録する
    const lastPosition = { x: null, y: null };
  
    // マウスがドラッグされているか(クリックされたままか)判断するためのフラグ
    let isDrag = false;
  
    // 絵を書く
    function draw(x, y) {
      // マウスがドラッグされていなかったら処理を中断する。
      // ドラッグしながらしか絵を書くことが出来ない。
      if(!isDrag) {
        return;
      }
  
      // 「context.beginPath()」と「context.closePath()」を都度draw関数内で実行するよりも、
      // 線の描き始め(dragStart関数)と線の描き終わり(dragEnd)で1回ずつ読んだほうがより綺麗に線画書ける
  
      // 線の状態を定義する
      // MDN CanvasRenderingContext2D: https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/lineJoin
      context.lineCap = 'round'; // 丸みを帯びた線にする
      context.lineJoin = 'round'; // 丸みを帯びた線にする
      context.lineWidth = 5; // 線の太さ
      context.strokeStyle = 'black'; // 線の色
  
      // 書き始めは lastPosition.x, lastPosition.y の値はnullとなっているため、
      // クリックしたところを開始点としている。
      // この関数(draw関数内)の最後の2行で lastPosition.xとlastPosition.yに
      // 現在のx, y座標を記録することで、次にマウスを動かした時に、
      // 前回の位置から現在のマウスの位置まで線を引くようになる。
      if (lastPosition.x === null || lastPosition.y === null) {
        // ドラッグ開始時の線の開始位置
        context.moveTo(x, y);
      } else {
        // ドラッグ中の線の開始位置
        context.moveTo(lastPosition.x, lastPosition.y);
      }
      // context.moveToで設定した位置から、context.lineToで設定した位置までの線を引く。
      // - 開始時はmoveToとlineToの値が同じであるためただの点となる。
      // - ドラッグ中はlastPosition変数で前回のマウス位置を記録しているため、
      //   前回の位置から現在の位置までの線(点のつながり)となる
      context.lineTo(x, y);
  
      // context.moveTo, context.lineToの値を元に実際に線を引く
      context.stroke();
  
      // 現在のマウス位置を記録して、次回線を書くときの開始点に使う
      lastPosition.x = x;
      lastPosition.y = y;
    }
  
    // canvas上に書いた絵を全部消す
    function clear() {
      context.clearRect(0, 0, canvas.width, canvas.height);
    }
  
    // マウスのドラッグを開始したらisDragのフラグをtrueにしてdraw関数内で
    // お絵かき処理が途中で止まらないようにする
    function dragStart(event) {
      // これから新しい線を書き始めることを宣言する
      // 一連の線を書く処理が終了したらdragEnd関数内のclosePathで終了を宣言する
      context.beginPath();
  
      isDrag = true;
    }
    // マウスのドラッグが終了したら、もしくはマウスがcanvas外に移動したら
    // isDragのフラグをfalseにしてdraw関数内でお絵かき処理が中断されるようにする
    function dragEnd(event) {
      // 線を書く処理の終了を宣言する
      context.closePath();
      isDrag = false;
  
      // 描画中に記録していた値をリセットする
      lastPosition.x = null;
      lastPosition.y = null;
    }
  
    // マウス操作やボタンクリック時のイベント処理を定義する
    function initEventHandler() {
      const clearButton = document.querySelector('#clear-button');
      clearButton.addEventListener('click', clear);
  
      canvas.addEventListener('mousedown', dragStart);
      canvas.addEventListener('mouseup', dragEnd);
      canvas.addEventListener('mouseout', dragEnd);
      canvas.addEventListener('mousemove', (event) => {
        // eventの中の値を見たい場合は以下のようにconsole.log(event)で、
        // デベロッパーツールのコンソールに出力させると良い
        // console.log(event);
  
        draw(event.layerX, event.layerY);
      });
    }
  
    // イベント処理を初期化する
    initEventHandler();

    //getImageTensor();
  });



  const CANVAS_SIZE = 280;
  const CANVAS_SCALE = 0.5;
  

  
  let isMouseDown = false;
  let hasIntroText = true;
  let lastX = 0;
  let lastY = 0;
  
  // Load our model.
  const sess = new onnx.InferenceSession();
  const loadingModelPromise = sess.loadModel("./onnx_model.onnx");
  






async function getImageTensor() {

    const session = new onnx.InferenceSession({ backendHint: 'webgl' })
    //console.log("sedd");
    // ONNX形式のモデルファイル
    const modelFile = 'mnist.onnx'
    //console.log("modelok");
    // モデルの読み込み
    await session.loadModel(modelFile)



    const ctx = document.getElementById('draw-area').getContext('2d')
     // input-canvasのデータを28x28に変換して、input-canvas-scaledに書き込む
    //const ctxScaled = document.getElementById('draw-area').getContext('2d')
    //ctxScaled.save();
    //ctxScaled.scale(28 / ctx.width, 28 / ctx.height)
    //ctxScaled.clearRect(0, 0, ctx.width, ctx.height)
    //ctxScaled.drawImage(document.getElementById('draw-area'), 0, 0)
    //ctxScaled.restore()

     // input-canvas-scaledのデータをTensorに変換
    //const imageDataScaled = ctxScaled.getImageData(0, 0, 28, 28)
    //const imgData = ctx.getImageData(0, 0, 400, 400);
    //const input = new onnx.Tensor(new Float32Array(imgData.data), "float32");
  
    const outputMap = await session.run([input]);
    const outputTensor = outputMap.values().next().value;
    const predictions = outputTensor.data;
    const maxPrediction = Math.max(...predictions);
    // console.log('imageDataScaled', imageDataScaled)

    // const input = new Float32Array(784);
    //for (let i = 0, len = imageDataScaled.data.length; i < len; i += 4) {
    //    input[i / 4] = imageDataScaled.data[i + 3] / 255;
    //}
    //const tensor = new onnx.Tensor(input, 'float32', [1, 1, 28, 28]);
    //console.log(tensor);
    //return tensor
    // Run model with Tensor inputs and get the result by output name defined in model.
  // Get the predictions for the canvas data.
    //const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    //const input = new onnx.Tensor(new Float32Array(imgData.data), "float32");



    // Check if result is expected.
    const prediction = document.getElementById('predictions');
    //console.log(outputMap);
    prediction.innerHTML = `final predict ${maxPrediction}`;

}


async function updatePredictions() {
    // Get the predictions for the canvas data.

    const ctx = document.getElementById('draw-area').getContext('2d')

    const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const input = new onnx.Tensor(new Float32Array(imgData.data), "float32");
  
    const outputMap = await sess.run([input]);
    const outputTensor = outputMap.values().next().value;
    const predictions = outputTensor.data;
    const maxPrediction = Math.max(...predictions);

    //console.log(maxPrediction);
    //console.log(predictions);
    index=0;
//    for (let i = 0; i < predictions.length; i++) {
  //      console.log(index,predictions[i]);
    //    index+=1;
    //console.log(predictions);   
    const predict = document.getElementById('predictions');
    predict.innerHTML = '';
    const results = [];

    max=0;
    for (let i = 0; i < predictions.length; i++) {
      results.push(`${index}: ${Math.round(100 * predictions[i])}%`);
      //console.log(results[i]);
      index+=1;
    }
/*
    for (let i = 0; i < predictions.length; i++) {
        if(max<predictions[i]){
            max=predictions[i];
        }
        index+=1; 
      }
      console.log(max);*/

    predict.innerHTML = results.join('<br/>');


}
 