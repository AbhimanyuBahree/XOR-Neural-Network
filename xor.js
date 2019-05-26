console.log("Started training");
$(function(){

$("#btn").click(function(event){
const no_epochs=document.getElementById("train-epochs").value;
const model=tf.sequential();
model.add(tf.layers.dense({units:4,activation:'relu',inputDim:2,kernelInitialier:'zeros'}));
model.add(tf.layers.dense({units:4,activation:'relu'}));
model.add(tf.layers.dense({units:1,activation:'sigmoid'}));

const learningRate=1;
const optimizer = tf.train.sgd(learningRate);
model.compile({optimizer:optimizer,loss:'binaryCrossentropy',metrics:['accuracy']});




const xs = tf.tensor([[0,0],[0,1],[1,0],[1,1]],[4,2]);
const ys = tf.tensor([0,1,1,0],[4,1]); 


let datab=[]; let datab2=[];
model.fit(xs, ys, {epochs: no_epochs,callbacks:{
onEpochEnd: async (epoch,logs) => { 
let f=epoch+1;
let x= (f/no_epochs * 100).toFixed(1);
document.getElementById("status").innerHTML="Training ...."+x+"%<br> Epochs: "+f+" Loss: "+logs.loss+" Accuracy: "+logs.acc;
datab.push({x:epoch,y:logs.loss});
datab2.push({x:epoch,y:logs.acc});
}
}}).then(() => {
let t=document.getElementById("root");
const data={values:datab,series:['Loss ']};
tfvis.render.linechart(t,data,{xLabel:"Epochs",yLabel:"Loss"});


let t2=document.getElementById("root2");
const data2={values:datab2,series:['Accuracy']};
tfvis.render.linechart(t2,data2,{xLabel:"Epochs",yLabel:"Accuracy"});
console.log("Done training. Evaluating model...");
const r = model.evaluate(xs, ys);
    console.log("Loss:");
    r[0].print();
    console.log("Accuracy:");
    r[1].print();
    
let  ru=model.predict(tf.tensor2d([0, 0], [1, 2])).asScalar();
let ru2=model.predict(tf.tensor2d([0, 1], [1, 2])).asScalar();
let ru3=model.predict(tf.tensor2d([1, 0], [1, 2])).asScalar();
let ru4=model.predict(tf.tensor2d([1, 1], [1, 2])).asScalar();

    document.getElementById("results").innerHTML="Testing 0,0<br>"+
     "<div id='ru'>"+ru+"</div>"+"<br>Testing 0,1<br>"+ru2+
    "<br>Testing 1,0<br>"+ru3+"<br>Testing 1,1<br>"+ru4;
  //let x= ru.array();
/*if(ru.toInt().asSc == tf.tensor(0,[1,1]).toInt().asScalar())
{
   document.getElementById("ru").style.color="green";
}*/
 
        
    console.log("Testing 0,0");
    model.predict(tf.tensor2d([0, 0], [1, 2])).print();
    
    console.log("Testing 0,1");
    model.predict(tf.tensor2d([0, 1], [1, 2])).print();
    console.log("Testing 1,0");
    model.predict(tf.tensor2d([1, 0], [1, 2])).print();
    console.log("Testing 1,1");
    model.predict(tf.tensor2d([1, 1], [1, 2])).print();
});


});
});
