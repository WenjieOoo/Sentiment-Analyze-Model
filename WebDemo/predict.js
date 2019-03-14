const MAXLEN=100
const START_WORD_INDEX = 1
const OOV_WORD_INDEX = 2
const INDEX_FROM = 3
var wordDict = {}
var dataReady = false
var currEmotion = -1
var wordIndexURL = "https://raw.githubusercontent.com/WenjieOoo/Sentiment-Analyze-Model-Data/master/imdb_dataset_word_index_top20000.json"     
var modelURL = "https://raw.githubusercontent.com/WenjieOoo/Sentiment-Analyze-Model-Data/master/Sentiment_Analyze_Model_1_8.bin"
var baiduTranslatorURL = 'https://api.fanyi.baidu.com/api/trans/vip/translate?'
var model;
var probindex = []  //概率结果顺序
var select = 0      //当前采用的表情标签值
var labels = ['恶心的', '开心的', '生气的', '一般般', '惊讶的', '悲伤的', '害怕的']
var flag = false;

/*加载模型*/
window.onload = function(){
    //$('body,html').animate({'scrollTop':'fast'});
    $.blockUI({ message: '<h1> 模型加载中 </h1>' });
    console.log("model reloading...")
    model = new KerasJS.Model({
        filepath: modelURL,
        gpu: false
    })
    axios.get(wordIndexURL).then(function(response) {
        wordDict = response.data;
    })
    model.ready().then(() => {
        $.unblockUI();
        console.log("model.ready...")
    })

    window.timer = window.setInterval(function(){
        var str = $('textarea').val();
        getPrediction(str);
        var result = sessionStorage.getItem("result");
        selectPicture(result);    
    },1000);
};

/*计数器*/
$('*').on("keyup","textarea",function(){
    var str = $(this).val();
    var length = str.length;
    var NewStr = length+"/50"; 
    $(this).next().text(NewStr);
})

/*根据表情选择对应的图片*/
function selectPicture(result){
    console.log("...sss");
    switch(result){
        case "恶心的":
        {
            $('#face').attr('src',"faceEmotion/disgust.png");
            break;
        }
        case "生气的":
        {
            $('#face').attr('src',"faceEmotion/angry.png");
            break;
        }
        case "惊讶的":
        {
            $('#face').attr('src',"faceEmotion/surprise.png");
            break;
        }
        case "悲伤的":
        {
            $('#face').attr('src',"faceEmotion/sad.png");
            break;
        }
        case "害怕的":
        {
            $('#face').attr('src',"faceEmotion/fear.png");
            break;
        }
        case "开心的":
        {
            $('#face').attr('src',"faceEmotion/happy.png");
            break;
        }
        default:
        {
            $('#face').attr('src',"faceEmotion/normal.png");
            break;
        }
    }
    $('#result p span').text(result);
}


function getPrediction(inputtext) {
    var input = new Float32Array(MAXLEN)
    // 调用翻译的过程，inputtext为要翻译的文字-------------------------------------------------------------------------------
    salt = 1435660288
    needMD5 = '20181112000233191' + inputtext + salt.toString() + 'JMuQLVm5_aRDvvRFoSqM'
    key = md5(needMD5);
    $.ajax({
        url:baiduTranslatorURL + 'q=' + inputtext + '&from=zh&to=en&appid=20181112000233191&salt=1435660288&sign=' + key,
        dataType:'jsonp',
        processData: false, 
        type:'get',
        // 调用翻译成功，则
        success:function(data){
            var inputTextParsed = data.trans_result[0].dst.trim().toLowerCase().split(/[\s.,!?]+/gi)
            let indices = inputTextParsed.map(word => {
                const index = wordDict[word]
                return !index ? OOV_WORD_INDEX : index + INDEX_FROM
            })
            indices = [START_WORD_INDEX].concat(indices)
            indices = indices.slice(-MAXLEN)
            const start = Math.max(0, MAXLEN - indices.length)
            for (let i = start; i < MAXLEN; i++) {
                input[i] = indices[i - start]
            }
            model.ready().then(() => {
                return model.predict({ input: input })
            }).then(outputData => {
                select = 0
                var probresult = new Float32Array(outputData.output)
                probresult = probresult.sort()
                for(let i = 0;i<probresult.length;i++){
                    probindex[i] = new Float32Array(outputData.output).indexOf(probresult[probresult.length-i-1])
                }
                var result = labels[probindex[select]];
                sessionStorage.setItem("result",result);
                console.log(result);
            })
        },
        // 若调用翻译失败
          error:function(XMLHttpRequest, textStatus, errorThrown) {
          alert(XMLHttpRequest.status);
          alert(XMLHttpRequest.readyState);
          alert(textStatus);
        }
    });

}