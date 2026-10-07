const $=id=>document.getElementById(id);let model,detector,stream,running=false,mode='live',busy=false;
const video=$('video'),photo=$('image'),canvas=$('overlay'),ctx=canvas.getContext('2d');
function message(text){$('message').textContent=text}
function reset(){ctx.clearRect(0,0,canvas.width,canvas.height);$('result').textContent='Waiting for input';$('confidence').textContent='—';$('bar').style.width='0%'}
function stop(){running=false;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;$('stop').disabled=true;$('start').disabled=false;$('status').textContent='Camera stopped'}
function tab(next){stop();mode=next;reset();video.hidden=true;photo.hidden=true;$('empty').hidden=false;$('live').classList.toggle('selected',next==='live');$('photo').classList.toggle('selected',next==='photo');$('start').hidden=next!=='live';$('stop').hidden=next!=='live';$('upload').hidden=next!=='photo';$('save').disabled=true;message(next==='live'?'Start your camera when ready.':'Choose a photo with a clearly visible face.')}
async function predict(source){if(busy)return;busy=true;try{const w=source.videoWidth||source.naturalWidth,h=source.videoHeight||source.naturalHeight;canvas.width=w;canvas.height=h;document.querySelector('.view').style.aspectRatio=`${w}/${h}`;const faces=await detector.estimateFaces(source,false);ctx.clearRect(0,0,w,h);let results=[];for(const face of faces){let [x,y]=face.topLeft,[x2,y2]=face.bottomRight;x=Math.max(0,Math.floor(x));y=Math.max(0,Math.floor(y));x2=Math.min(w,Math.ceil(x2));y2=Math.min(h,Math.ceil(y2));if(x2<=x||y2<=y)continue;const p=tf.tidy(()=>{const pixels=tf.browser.fromPixels(source);const crop=pixels.slice([y,x,0],[y2-y,x2-x,3]);const input=tf.image.resizeBilinear(crop,[160,160]).toFloat().div(127.5).sub(1).expandDims();return model.predict(input).dataSync()[0]});const label=p>=.5?'No Mask':'Mask',confidence=p>=.5?p:1-p;results.push({label,confidence});ctx.strokeStyle=p>=.5?'#ff736e':'#45e1ad';ctx.lineWidth=Math.max(2,w/200);ctx.strokeRect(x,y,x2-x,y2-y);ctx.font=`bold ${Math.max(16,w/35)}px Arial`;ctx.fillStyle=ctx.strokeStyle;ctx.fillText(`${label} ${(confidence*100).toFixed(1)}%`,x,Math.max(24,y-9))}if(results.length){const r=results[0];$('result').textContent=r.label;$('confidence').textContent=`${(r.confidence*100).toFixed(1)}%`;$('bar').style.width=`${r.confidence*100}%`;$('detail').textContent=results.length>1?`${results.length} faces detected. Confidence shown for the first face.`:'Prediction for the detected face.'}else{$('result').textContent='No face detected';$('detail').textContent='Face forward and move closer in good light.';$('confidence').textContent='—';$('bar').style.width='0%'}$('save').disabled=false}catch(e){message('Prediction failed: '+e.message)}finally{busy=false}}
async function loop(){if(!running)return;if(model&&detector&&video.readyState>=2)await predict(video);if(running)setTimeout(loop,180)}
$('start').onclick=async()=>{
  $('start').disabled=true;
  message('Requesting camera access. Choose Allow in your browser.');
  try{
    if(!window.isSecureContext)throw new Error('Open the HTTPS website to use the camera.');
    if(!navigator.mediaDevices?.getUserMedia)throw new Error('Open this website directly in Chrome or Edge to use your camera.');
    stream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
    video.srcObject=stream;video.hidden=false;photo.hidden=true;$('empty').hidden=true;
    await video.play();running=true;$('stop').disabled=false;$('status').textContent=model&&detector?'Camera live · Detection ready':'Camera live · Detection loading';
    message(model&&detector?'Try with and without a mask.':(loadError?'Detection could not load: '+loadError:'Camera is ready. Detection models are loading…'));loop();
  }catch(e){
    stop();video.hidden=true;$('empty').hidden=false;
    const tips={NotAllowedError:'Camera access was blocked. Open this link directly in Chrome or Edge, then allow Camera in the address-bar site settings and try again.',NotFoundError:'No camera was found. Connect a webcam and try again.',NotReadableError:'The camera is busy or unavailable. Close the Python webcam demo, Teams, Zoom, and other camera apps, then try again.',OverconstrainedError:'Your camera does not support the requested settings.',SecurityError:'Camera access is restricted. Open the website directly in Chrome or Edge.'};
    message(tips[e.name]||'Camera could not start: '+e.message);
  }
};
$('stop').onclick=()=>{stop();message('Camera stopped. Your camera is no longer in use.')};$('live').onclick=()=>tab('live');$('photo').onclick=()=>tab('photo');
$('file').onchange=async e=>{const file=e.target.files[0];if(!file)return;if(!model||!detector){message('Detection is still loading. Check the model status below.');return}const url=URL.createObjectURL(file);photo.onload=async()=>{photo.hidden=false;$('empty').hidden=true;$('status').textContent='Checking image…';await predict(photo);$('status').textContent='Image checked';URL.revokeObjectURL(url);message('Choose another image or save this result.')};photo.onerror=()=>{URL.revokeObjectURL(url);message('This image could not be opened. Try a JPG or PNG.')};photo.src=url};
$('save').onclick=()=>{const output=document.createElement('canvas');output.width=canvas.width;output.height=canvas.height;const c=output.getContext('2d');c.drawImage(mode==='live'?video:photo,0,0,output.width,output.height);c.drawImage(canvas,0,0);output.toBlob(blob=>{if(!blob)return;const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download='masklab-result.png';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)})};
window.addEventListener('pagehide',stop);
let loadError='';
function deadline(promise,label){return Promise.race([promise,new Promise((_,reject)=>setTimeout(()=>reject(new Error(label+' took too long. Reload to retry.')),45000))])}
(async()=>{try{
  $('result').textContent='Loading detection…';
  await tf.ready();
  $('status').textContent='Loading mask model…';
  model=await deadline(tf.loadLayersModel('./model/model.json'),'Mask model loading');
  $('status').textContent='Loading face detector…';
  detector=await deadline(blazeface.load({modelUrl:'./face-model/model.json'}),'Face detector loading');
  $('result').textContent='Waiting for input';
  $('status').textContent=running?'Camera live · Detection ready':'Model ready';
  message(running?'Model ready. Try with and without a mask.':'Ready. Start your camera or choose an image.');
}catch(e){loadError=e.message;$('status').textContent='Detection unavailable';$('result').textContent='Detection unavailable';$('detail').textContent='Reload the page to retry model loading.';message('Detection could not load: '+loadError)}})();
