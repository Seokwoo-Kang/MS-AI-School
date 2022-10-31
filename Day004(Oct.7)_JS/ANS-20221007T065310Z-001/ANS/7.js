function countKorean(array){
    let result = 0
    for(let i=0;i<array.length;i++){
        if(array[i] == '국어'){
            result += 1
        }
    }
    return result
}

let subs = ['국어','수학','영어','국어','과학']
let result = countKorean(subs)
console.log(result)
