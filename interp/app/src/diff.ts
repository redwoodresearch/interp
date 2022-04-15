// Given zero-based half-open range [start, end) of array indexes,
// return one-based closed range [start + 1, end] as string.
import diff from "diff-sequences"
const getRange = (start:number, end:number) =>
  start + 1 === end ? `${start + 1}` : `${start + 1},${end}`;

// Given index intervals of lines to delete or insert, or both, or neither,
// push formatted diff lines onto array.
const pushDelIns = (aLines:any[], aIndex:number, aEnd:number, bLines:any[], bIndex:number, bEnd:number, array:any[]) => {
  const deleteLines = aIndex !== aEnd;
  const insertLines = bIndex !== bEnd;
  const changeLines = deleteLines && insertLines;
  if (changeLines) {
    array.push(getRange(aIndex, aEnd) + 'c' + getRange(bIndex, bEnd));
  } else if (deleteLines) {
    array.push(getRange(aIndex, aEnd) + 'd' + String(bIndex));
  } else if (insertLines) {
    array.push(String(aIndex) + 'a' + getRange(bIndex, bEnd));
  } else {
    return;
  }

  for (; aIndex !== aEnd; aIndex += 1) {
    array.push('< ' + aLines[aIndex]); // delete is less than
  }

  if (changeLines) {
    array.push('---');
  }

  for (; bIndex !== bEnd; bIndex += 1) {
    array.push('> ' + bLines[bIndex]); // insert is greater than
  }
};

// Given content of two files, return emulated output of diff utility.
export const findShortestEditScript = (a:string, b:string) => {
  const aLines = a.split('\n');
  const bLines = b.split('\n');
  const aLength = aLines.length;
  const bLength = bLines.length;

  const isCommon = (aIndex:number, bIndex:number) => aLines[aIndex] === bLines[bIndex];

  let aIndex = 0;
  let bIndex = 0;
  const array = [] as any[];
  const foundSubsequence = (nCommon:number, aCommon:number, bCommon:number) => {
    pushDelIns(aLines, aIndex, aCommon, bLines, bIndex, bCommon, array);
    aIndex = aCommon + nCommon; // number of lines compared in a
    bIndex = bCommon + nCommon; // number of lines compared in b
  };

  diff(aLength, bLength, isCommon, foundSubsequence);

  // After the last common subsequence, push remaining change lines.
  pushDelIns(aLines, aIndex, aLength, bLines, bIndex, bLength, array);

  return array.length === 0 ? '' : array.join('\n') + '\n';
};

export function getIndexMapperOfDiff(prevToks:string[],newToks:string[]){
    const foundSubsequences = [] as [number,number,number][]
    diff(prevToks.length,newToks.length,(a,b)=>prevToks[a]===newToks[b],(a,b,c)=>foundSubsequences.push([a,b,c]))
    console.log({prevToks,newToks,foundSubsequences})
    return (aIndex:number)=>{
        for (let [n,a,b] of foundSubsequences){
            if (aIndex>=a && aIndex<n+a){
                return aIndex-a+b
            }
        }
        return null
    }
}
