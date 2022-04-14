import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import { range } from "./common";
import { ViewComponentProps, VeryNamedTensor } from "./proto";
import { useState } from 'react';
import ndarray from "ndarray";


interface PathToken {
    name: string,
    index: number;
}
interface Path {
    value: number,
    tokens: PathToken[];
}

function comparePaths(a: Path, b: Path): number {
    return a.value - b.value;
}

type Bucket = Path[];

function roundNumber(num: number): number {
    return (Math.round(num * 100) / 100);
}

function getBuckets(paths: Path[], numBuckets: number = 30, numPathsRendered: number = 20): { buckets: Path[][], bucketNames: string[], bucketLengths: number[]; } {
    paths.sort(comparePaths);
    const minValue: number = paths[0].value;
    const maxValue: number = paths[paths.length - 1].value;
    if (minValue === maxValue) {
        throw "minValue and maxValue are equal";
    }
    const bucketSize: number = (maxValue - minValue) / numBuckets;
    const buckets: Bucket[] = [];
    let currentBucket: Bucket = [];
    let currentBucketLowerBound: number = minValue;

    let bucketNames: string[] = [];
    let bucketLengths: number[] = [];
    paths.push({ value: maxValue + 0.5 * bucketSize, tokens: [{ name: 'dummy', index: 0 }] });
    for (let i = 0; i < paths.length; i++) {
        while (paths[i].value > (currentBucketLowerBound + bucketSize)) {
            bucketLengths.push(currentBucket.length);
            currentBucket.sort(() => Math.random() - 0.5);
            let renderedPaths: Path[] = currentBucket.slice(0, numPathsRendered);
            renderedPaths.sort(comparePaths);
            renderedPaths.reverse();

            buckets.push([...renderedPaths]);
            bucketNames.push(roundNumber(currentBucketLowerBound) + ' to ' + roundNumber(currentBucketLowerBound + bucketSize));
            currentBucketLowerBound += bucketSize;
            currentBucket = [];

            if (i === paths.length - 1) {
                currentBucket.push(paths[i]);
            }
        }
        currentBucket.push(paths[i]);
    }



    return { buckets, bucketNames, bucketLengths };
}

const getNumString = (num: number): string => {
    if (num === 0.0) {
        return '0.00';
    }
    if (num.toString()[0] === '-') {
        return '-' + (num.toString().slice(1) + "00").slice(0, 4);
    } else {
        return (num.toString() + "00").slice(0, 4);
    }
};

function renderBucketPaths(bucket: Bucket, getOnHoverFromTokenIdx: ((i: number) => (() => void))): JSX.Element {
    function renderPath(path: Path): JSX.Element {
        let pathComponentList: JSX.Element[] = path.tokens.map((token: PathToken) => [<span style={{ border: "1px solid black", display: "inline-block" }} onMouseOver={getOnHoverFromTokenIdx(token.index)} onMouseLeave={getOnHoverFromTokenIdx(-1)}>{token.name}</span>, <>→</>]).flat(1);
        pathComponentList = pathComponentList.slice(0, pathComponentList.length - 1);
        let pathComponent: JSX.Element = <>{getNumString(path.value) + ': '}{pathComponentList}</>;

        return pathComponent;
    }
    const bucketComponent: JSX.Element = <span style={{ whiteSpace: "pre-wrap" }}>{bucket.map((path: Path) => <div style={{ paddingBottom: ".2rem" }}>{renderPath(path)}</div>)}</span>;
    return bucketComponent;
}

function getPaths(tensor: ndarray.NdArray, dimIdxNames: string[][], dimTypes: string[], currentPrefix: PathToken[], includeSameTokenPaths: boolean): Path[] {
    const shape: number[] = tensor.shape;

    function checkIfPathPassesThroughDistinctTokens(path: Path): boolean {
        let pathTokens = path.tokens;
        let tokenIdxs = pathTokens.map((tok: PathToken) => tok.index);
        tokenIdxs = tokenIdxs.filter((idx: number) => (idx !== 0 && idx !== -1));
        const allEqual = (arr: any[]) => arr.every((v: any) => v === arr[0]);
        return (!allEqual(tokenIdxs) && (tokenIdxs.length > 0));
    }

    if (shape.length === 1) {
        let paths = range(shape[0]).map((i: number): Path => { return { value: roundNumber(tensor.get(i)), tokens: [...currentPrefix, { name: dimIdxNames[0][i], index: ((dimTypes[0] === 'seq') ? i : -1) }] }; });
        paths = paths.filter(path => !(isNaN(path.value) || path.value == null));
        return paths;
    } else {
        let paths = range(shape[0]).map(i => getPaths(tensor.pick(i), dimIdxNames.slice(1), dimTypes.slice(1), [...currentPrefix, { name: dimIdxNames[0][i], index: ((dimTypes[0] === 'seq') ? i : -1) }], includeSameTokenPaths)).flat(1);
        paths = paths.filter(path => !(isNaN(path.value) || path.value == null));
        if (!includeSameTokenPaths) {
            paths = paths.filter(path => checkIfPathPassesThroughDistinctTokens(path));
        }
        return paths;
    }
}

const getTokens = (vnt: VeryNamedTensor): string[] => {
    // check that seq is in dim_types
    const seq_idx = vnt.dim_types.indexOf('seq');
    let tokens = range(vnt.dim_idx_names[seq_idx].length).map((i: number) => vnt.dim_idx_names[seq_idx][i]);
    return tokens;
};



const TokenHighlighter = ({ tokens, highlightedTokenIdx }: { tokens: string[], highlightedTokenIdx: number; }): JSX.Element => {
    const getBackgroundColorOfIdx = (i: number) => { return (i === highlightedTokenIdx) ? 'rgba(255, 0,0, .3)' : 'white'; };

    // check that seq is in dim_types
    let result = tokens.map((tok: string, i: number): JSX.Element => <span key={i} style={{ border: "1px solid black", display: "inline-block", backgroundColor: getBackgroundColorOfIdx(i) }}>{tok}</span>);
    return <span style={{ whiteSpace: "pre-wrap" }}>{result}</span>;
};

interface HistogramSubcomponentProps {
    buckets: Bucket[],
    bucketNames: string[],
    bucketLengths: number[],
    tokens: string[];
}

function HistogramSubcomponent({ buckets, bucketNames, bucketLengths, tokens }: HistogramSubcomponentProps): JSX.Element {
    let [focusedBucketIdx, setFocusedBucketIndex] = useState(buckets.length - 1);

    let [highlightedTokenIdx, setHighlightedTokenIdx] = useState(-1);
    let getOnHoverFromTokenIdx = (i: number): (() => void) => { return function () { setHighlightedTokenIdx(i); }; };

    let options: object = {
        chart: {
            type: 'bar'
        },
        title: {
            text: 'Path Values'
        },
        subtitle: {
            text: 'Counts of path values in each bucket'
        },
        xAxis: {
            categories: bucketNames,
            title: {
                text: null
            }
        },
        yAxis: {
            min: 0,
            title: {
                text: 'Count',
                align: 'high'
            },
            labels: {
                overflow: 'justify'
            }
        },
        tooltip: {
            valueSuffix: ''
        },
        plotOptions: {
            bar: {
                dataLabels: {
                    enabled: true
                }
            },
            series: {
                allowPointSelect: true,
                point: {
                    events: {
                        select: (event: any) => {
                            let index: number = bucketNames.findIndex((bucketName: string) => (bucketName === event.target.category));
                            setFocusedBucketIndex(index);
                        }
                    }
                }

            }
        },
        legend: {
            layout: 'vertical',
            align: 'right',
            verticalAlign: 'top',
            x: -40,
            y: 80,
            floating: true,
            borderWidth: 1,
            backgroundColor: '#FFFFFF',
            shadow: true
        },
        credits: {
            enabled: false
        },
        series: [{
            showInLegend: false,
            name: 'Bucket Counts',
            data: bucketLengths
        }]
    };

    return <>
        <div>
            <HighchartsReact
                highcharts={Highcharts}
                options={options}
            />
        </div>
        <div>
            <TokenHighlighter tokens={tokens} highlightedTokenIdx={highlightedTokenIdx} />
        </div>
        <br />
        <>{renderBucketPaths(buckets[focusedBucketIdx], getOnHoverFromTokenIdx)}</>
    </>;
}

export default function InteractiveHistogram(props: ViewComponentProps): JSX.Element {
    const vnt = props.vnt;
    const [includeSameTokenPaths, toggleSameTokenPaths] = useState(false);
    const paths = getPaths(vnt.tensor, vnt.dim_idx_names, vnt.dim_types, [], includeSameTokenPaths);
    const sameTokensCheckbox = <div>
        <input type="checkbox" checked={includeSameTokenPaths} onChange={e => toggleSameTokenPaths(e.target.checked)}></input>
        Include same-token paths
    </div>;
    if (paths.length === 0) {
        return <div>
            {sameTokensCheckbox}
            <br></br>
            {"There are no valid paths under your settings."}
        </div>;
    }

    let { buckets, bucketNames, bucketLengths } = getBuckets(paths, 20);
    let tokens = getTokens(vnt);

    return <div>
        {sameTokensCheckbox}

        <HistogramSubcomponent buckets={buckets} bucketNames={bucketNames} bucketLengths={bucketLengths} tokens={tokens}></HistogramSubcomponent>
    </div>;
}