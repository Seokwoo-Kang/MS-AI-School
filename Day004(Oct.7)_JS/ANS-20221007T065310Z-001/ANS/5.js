function objectSum(numO){
	let result = 0
    for(let i = 0; i < numO.length; i++){
        if(numO[i]['number'] % 2 == 0){
            result += numO[i]['number']
        }
    }
    return result
}

let numObject = [{'name':'lee', 'number':22}, {'name':'park','number':11}]
let result = objectSum(numObject)
console.log(result)
