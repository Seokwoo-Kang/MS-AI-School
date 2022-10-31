function countSubject(subject,array){
    let result = 0
    for(let i=0;i<array.length;i++){
        if(array[i] == subject){
            result += 1
        }
    }
    return result
}

let subs = ['국어','수학','영어','국어','과학']
let result = countSubject('수학', subs)
console.log(result)
