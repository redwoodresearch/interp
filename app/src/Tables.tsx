import { WordPredictionPercent, WordPrediction } from "./proto";

export function PercentagesTable({ pcts }: { pcts: Array<WordPredictionPercent> }) {
    function makeRow({ word, pct }: WordPredictionPercent, index: number) {
        return (<tr key={index}><td>{word}</td><td>{pct.toFixed(2)}</td></tr>);
    }
    return (
        <table>
            <thead>
                <tr>
                    <th>Prediction</th>
                    <th>%</th>
                </tr>
            </thead>
            <tbody>
                {pcts.map(makeRow)}
            </tbody>
        </table>
    )
}


export function LogProbsTable({ logProbs }: { logProbs: Array<WordPrediction> }) {
    function makeRow({ word, logProb }: WordPrediction, index: number) {
        return (<tr key={index}><td>{word}</td><td>{logProb.toFixed(4)}</td></tr>);
    }
    return (
        <table>
            <thead>
                <tr>
                    <th>Prediction</th>
                    <th>Log-Probability</th>
                </tr>
            </thead>
            <tbody>
                {logProbs.map(makeRow)}
            </tbody>
        </table>
    )
}