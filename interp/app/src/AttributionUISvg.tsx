import { isOneArrayPrefixOfOther, maskToInverseIndex, range, stateToFakeTree } from "./common";
import { AttributionPath, AttributionRoot, AttributionStateSpec, AttributionTensors, AttributionTreeTensors, AttribLocation, LogitRoot, PickArgs, Attributions, LocationUsedAndBase, AttributionTreeSpec } from "./proto";
import { colorNegPos, tokenToInlineString } from "./ui_common";

export const attRootToString = (ar: AttributionRoot) => {
    // poor man's case statement - I have to check kind and cast myself :(
    if (ar.kind === "logprob") {
        const root = ar.data as LogitRoot;
        return `logprob of ${tokenToInlineString(root.tokString)} ${root.comparisonTokString ? ` vs ${tokenToInlineString(root.comparisonTokString)}` : ''}`;
    } else if (ar.kind === "logit") {
        const root = ar.data as LogitRoot;
        return `logit of ${tokenToInlineString(root.tokString)} ${root.comparisonTokString ? ` vs ${tokenToInlineString(root.comparisonTokString)}` : ''}`;
    } else if (ar.kind === "prob") {
        const root = ar.data as LogitRoot;
        return `prob of ${tokenToInlineString(root.tokString)} ${root.comparisonTokString ? ` vs ${tokenToInlineString(root.comparisonTokString)}` : ''}`;
    } else if (ar.kind === "attention_pattern") {
        throw new Error(`not supported yet ${ar}`);
    } else if (ar.kind === "attention_pattern_from") {
        throw new Error(`not supported yet ${ar}`);
    } else if (ar.kind === "attention_single") {
        throw new Error(`not supported yet ${ar}`);
    } else {
        throw new Error(`invalid attribution root kind ${ar}`);
    }
};

export const getTreePath: (state: AttributionTensors | AttributionStateSpec, path: AttributionPath, layerNames: any) => AttributionTreeSpec | AttributionTreeTensors =
    (state: AttributionTensors | AttributionStateSpec, path: AttributionPath, layerNames: any) => {
        let newTree = stateToFakeTree(state, layerNames);
        for (let p of path) {
            console.log({ p, t: newTree.children });
            newTree = newTree.children.filter((x) => JSON.stringify(x.idx) === JSON.stringify(p))[0];
        }
        return newTree;
    };

export const getPathAttributionValues = (tree: AttributionTreeTensors, path: AttributionPath) => {
    if (path.length === 0) {
        return [];
    }
    const currentAttrib = indexAttributionsByLocation(tree.attribution, path[0]);
    const nextTree = tree.children.filter((x) => JSON.stringify(x.idx) === JSON.stringify(path[0]))[0];
    const inner = getPathAttributionValues(nextTree, path.slice(1)) as number[];
    return [currentAttrib, ...inner];
};

export const indexAttributionsByLocation = (attrib: Attributions, location: AttribLocation) => {
    if (location.layerWithIO === 0) {
        return attrib.embeds.tensor.get(location.token);
    } else if (location.isMlp) {
        return attrib.mlps.tensor.get(location.layerWithIO - 1, location.headOrNeuron, location.token);
    } else if (!location.isMlp) {
        return attrib.heads.tensor.get(location.layerWithIO - 1, location.headOrNeuron, location.token);
    } else {
        throw Error("hi");
    }
};

const isOnePathPrefixOfOtherIgnoringQkvEnd = (path1: AttributionPath, path2: AttributionPath) => {
    if (path1.length === 0 || path2.length === 0) return true;
    path1 = path1.map(x => ({ token: x.token, layerWithIO: x.layerWithIO, headOrNeuron: x.headOrNeuron, qkv: x.qkv, isMlp: x.isMlp }));
    path2 = path2.map(x => ({ token: x.token, layerWithIO: x.layerWithIO, headOrNeuron: x.headOrNeuron, qkv: x.qkv, isMlp: x.isMlp }));
    if (path1.length > path2.length) {
        path1[path2.length - 1] = { ...path1[path2.length - 1], qkv: path2[path2.length - 1].qkv };
    } else {
        path2[path1.length - 1] = { ...path2[path1.length - 1], qkv: path1[path1.length - 1].qkv };
    }
    return isOneArrayPrefixOfOther(path1, path2);
};

export function AttributionUISvg(props: { state: AttributionTensors, ohInner: (...x: any[]) => void, layerNamesWithIO: string[], tokens: string[], hoveredPath: AttributionPath | null, stateSpec: AttributionStateSpec; availableWidth: number; availableHeight: number; }) {
    const { state, ohInner, tokens, layerNamesWithIO, hoveredPath, stateSpec, availableWidth, availableHeight } = props;

    console.log("svg props", props);
    const c = {
        topAxisMargin: 4,

        maxWordWidth: 50,
        fontSize: 12,

        seperation: 50,
        headMlpVerticalSeperation: 50,
        headWidth: 34,
        wordMargin: 3,
        smallFontSize: 8,
        headSpacing: 1,
        conjoinedHeadSpacing: 0,
        seqSpacing: 20,
        headHeight: 30,
        neuronWidth: 24,
        neuronHeight: 10,
        qkvStride: 11,
        qkvPermutation: [2, 0, 1], // show them in this order because k goes to early tokens, q goes to later tokens

        lineWidthMax: 3,
        lineWidthOnHover: 5,
        lineWidthOnUnHover: 0.2,
        backgroundOpacityFactor: 0.1,
        splineSharpness: 0.7,
        hueNeg: 0.6,
        huePos: 0.0,
        lineHoverMouseWidth: 7,
        useSplines: true,
        headHoveredStrokeWidth: 4,
    };

    const usedLayersMask = layerNamesWithIO.map((x, i) => (i === 0 || i === layerNamesWithIO.length - 1 ? [true, false] : [false, false]) as [boolean, boolean]);// always show input layer
    const usedTokensMask = tokens.map(x => false);
    const fakeTree = stateToFakeTree(state, layerNamesWithIO);
    const recursivelyFillInUsedTokensAndLayers = (tree: AttributionTreeTensors) => {
        usedLayersMask[tree.idx.layerWithIO][tree.idx.isMlp ? 1 : 0] = true;
        usedTokensMask[tree.idx.token] = true;
        if (tree.outgoingLines) {
            tree.outgoingLines.forEach(idx => {
                usedLayersMask[idx.base.layerWithIO][idx.base.isMlp ? 1 : 0] = true;
                usedTokensMask[idx.base.token] = true;
            });
        }
        tree.children.map(recursivelyFillInUsedTokensAndLayers);
    };
    recursivelyFillInUsedTokensAndLayers(fakeTree);

    const usedLayers = range(usedLayersMask.length).filter(x => usedLayersMask[x].some(x => x));
    const n_used_layers = usedLayers.length;

    const usedTokens = range(usedTokensMask.length).filter(x => usedTokensMask[x]);
    console.log("usedTokens", usedTokens);
    const usedHeads = usedLayers.map((layer_idx, used_layer_idx) => usedTokens.map((tok_idx, used_tok_idx) =>
        ([] as { headOrNeuron: number, paths: AttributionPath[]; }[])
    ));

    const usedNeurons = usedLayers.map((layer_idx, used_layer_idx) => usedTokens.map((tok_idx, used_tok_idx) =>
        ([] as { headOrNeuron: number, paths: AttributionPath[]; }[])
    ));

    const layerIdxsToUsed = maskToInverseIndex(usedLayersMask.map(x => x.some(y => y)));
    const tokenIdxsToUsed = maskToInverseIndex(usedTokensMask);

    const conjoinedHeadsToPxWidth = (nconjoineds: number[]) => {
        let val = c.seqSpacing + (nconjoineds.length - 1) * c.headSpacing + (nconjoineds.reduce((a, b) => a + b, 0)) * (c.headWidth + c.conjoinedHeadSpacing);
        return Math.max(c.maxWordWidth, val);
    };
    const conjoinedNeuronsToPxWidth = (neurons: number[]) => {
        let val = c.seqSpacing + (neurons.reduce((a, b) => a + b, 0)) * c.neuronWidth;
        return Math.max(c.maxWordWidth, val);
    };

    const seqMaxWidths = usedTokens.map(() => 0);

    const fillUsedHeadsAndNeurons = (idx: AttribLocation, pastPath: AttributionPath) => {

        const hereUsed = { layerWithIO: layerIdxsToUsed[idx.layerWithIO], token: tokenIdxsToUsed[idx.token], isMlp: idx.isMlp, headOrNeuron: -1, headReplica: -1, qkv: idx.qkv };
        // console.log("hereUsed", hereUsed);
        const hereHeads = usedHeads[hereUsed.layerWithIO][hereUsed.token];
        const hereNeurons = usedNeurons[hereUsed.layerWithIO][hereUsed.token];
        const hereHeadOrNeurons = hereUsed.isMlp ? hereNeurons : hereHeads;
        let matchingHeadIdx = range(hereHeadOrNeurons.length).filter(i => hereHeadOrNeurons[i].headOrNeuron === idx.headOrNeuron)[0];
        if (matchingHeadIdx === undefined) {
            hereHeadOrNeurons.push({ headOrNeuron: idx.headOrNeuron, paths: [pastPath] });
            matchingHeadIdx = hereHeadOrNeurons.length - 1;
            hereUsed.headReplica = 0;
        } else {
            const matchingHeadOrNeuron = hereHeadOrNeurons[matchingHeadIdx];
            hereUsed.headReplica = range(matchingHeadOrNeuron.paths.length).find(i => JSON.stringify(matchingHeadOrNeuron.paths[i]) === JSON.stringify(pastPath)) as number;
            if (hereUsed.headReplica === undefined) {
                hereUsed.headReplica = matchingHeadOrNeuron.paths.length;
                matchingHeadOrNeuron.paths.push(pastPath);
            }
        }
        hereUsed.headOrNeuron = matchingHeadIdx;

        seqMaxWidths[hereUsed.token] = Math.max(seqMaxWidths[hereUsed.token], conjoinedHeadsToPxWidth(hereHeads.map(x => x.paths.length)), conjoinedNeuronsToPxWidth(hereNeurons.map(x => x.paths.length)));
        return hereUsed;
    };

    const recursivelyGetUsedHeadsAndNeurons = (tree: AttributionTreeTensors, pastPath: AttributionPath) => {
        const path = [...pastPath, tree.idx];
        if (tree.idx.layerWithIO === layerNamesWithIO.length - 1) path.pop();
        tree.usedIdx = fillUsedHeadsAndNeurons(tree.idx, pastPath);
        tree.outgoingLines?.forEach(usedAndBase => {
            usedAndBase.used = fillUsedHeadsAndNeurons(usedAndBase.base, path);
        });
        tree.children.map((x) => recursivelyGetUsedHeadsAndNeurons(x, path));
    };

    recursivelyGetUsedHeadsAndNeurons(fakeTree, []);

    const seqPoses = [] as number[];
    let curxpos = 0;

    for (let i = 0; i < seqMaxWidths.length; i++) {
        seqPoses.push(curxpos + seqMaxWidths[i] / 2);
        curxpos += seqMaxWidths[i];
    }
    seqPoses.push(curxpos);

    let curypos = 0;
    const layerPoses = [] as [number, number][];
    for (let i = 0; i < usedLayersMask.length; i++) {
        const hereHeadMlp = usedLayersMask[i];
        if (hereHeadMlp[0] || hereHeadMlp[1]) {
            const hereHeadMlpPos = [-1, -1] as [number, number];
            layerPoses.push(hereHeadMlpPos);
            if (hereHeadMlp[0]) {
                hereHeadMlpPos[0] = curypos;
                curypos += c.headHeight;
            }
            if (hereHeadMlp[1]) {
                if (hereHeadMlp[0]) {
                    curypos += c.headMlpVerticalSeperation;
                }
                hereHeadMlpPos[1] = curypos;
                curypos += c.neuronHeight;
            }
            curypos += c.seperation;
        }
    }
    layerPoses.push([curypos, curypos]);

    // currently just putting words over where they should go
    // and not worrying about their widths
    const yTop = (c.fontSize + c.wordMargin) * 2 + c.headHeight + c.topAxisMargin;
    const neededHeight = yTop + Math.max(...layerPoses[layerPoses.length - 1]);
    const neededWidth = (seqPoses[seqPoses.length - 1]) + c.maxWordWidth * 2;
    console.log("SVG pre render state", { state, usedHeads, seqPoses, seqMaxWidths, layerPoses, usedTokens, usedNeurons, usedLayersMask, usedLayers });
    // const inputSums = state.tree.length > 0 && covTreeSumPick(state.tree, [0, 0, null, 0]);

    const isLocationHovered = (hoveredPath: AttributionPath | null, location: AttribLocation, qkv: boolean = false) => {
        if (hoveredPath === null) return false;
        for (let pathLoc of hoveredPath) {
            if (pathLoc.token === location.token && pathLoc.layerWithIO === location.layerWithIO && pathLoc.headOrNeuron === location.headOrNeuron && pathLoc.isMlp === location.isMlp && (!qkv || location.qkv == pathLoc.qkv)) {
                return true;
            }
        }
        return false;
    }; // TODO: make this work for partial locations

    const usedLocToYPos = (usedLocation: AttribLocation) => {
        return layerPoses[usedLocation.layerWithIO][usedLocation.isMlp ? 1 : 0];
    };


    const usedLocToXPos = (usedLocation: AttribLocation) => {
        // console.log("usedLocToXPos", usedLocation)
        let x = seqPoses[usedLocation.token];
        if (usedLayers[usedLocation.layerWithIO] == 0) {
            const paths = usedHeads[0][usedLocation.token][0]?.paths;
            if (paths !== undefined) {
                if (usedLocation.headReplica !== undefined) {
                    x -= (paths.length - 1) * c.neuronWidth / 2;
                    x += usedLocation.headReplica * c.neuronWidth;
                }
            }
        } else if (usedLayers[usedLocation.layerWithIO] < layerNamesWithIO.length - 1) {
            x += 11; // HACK I don't know why I need this
            if (usedLocation.isMlp) {
                const layerSeqNeurons = usedNeurons[usedLocation.layerWithIO][usedLocation.token];
                x -= layerSeqNeurons.map(x => x.paths.length).reduce((a, b) => a + b, 0) / 2 * c.neuronWidth;
                x += layerSeqNeurons.slice(0, usedLocation.headOrNeuron).map(x => x.paths.length).reduce((a, b) => a + b, 0) * c.neuronWidth;
                if (usedLocation.headReplica !== undefined) {
                    x += usedLocation.headReplica * c.neuronWidth;
                } else {
                    console.error("no head replica!", usedLocation);
                }
            } else {
                const layerSeqHeads = usedHeads[usedLocation.layerWithIO][usedLocation.token];
                x -= (conjoinedHeadsToPxWidth(layerSeqHeads.map(x => x.paths.length)) - c.headWidth) / 2;
                const prevConjoinedHeadsWidth = layerSeqHeads.slice(0, usedLocation.headOrNeuron).reduce((acc, val) => acc + val.paths.length * (c.headWidth + c.conjoinedHeadSpacing) + c.headSpacing, 0);
                x += prevConjoinedHeadsWidth;
                if (usedLocation.headReplica === undefined) {
                    throw Error('no head replica');
                }
                x += usedLocation.headReplica * c.headWidth;
                if (usedLocation.qkv !== null && usedLocation.qkv !== undefined) {
                    x += (c.qkvPermutation[usedLocation.qkv] - 1) * c.qkvStride;
                }
            }
        }
        return x;
    };

    const heads = [] as JSX.Element[];

    usedHeads.forEach((layer, usedLayerIdx) => {
        if (usedLayers[usedLayerIdx] === 0 || usedLayers[usedLayerIdx] === layerNamesWithIO.length - 1) {
            return;
        }
        layer.forEach((seq, usedSeqIdx) => {
            seq.forEach((head, usedHeadIdx) => {
                const avg = 0;
                const layerIdx = usedLayers[usedLayerIdx];
                const seqIdx = usedTokens[usedSeqIdx];
                const headIdx = head.headOrNeuron;
                const loc = { layerWithIO: layerIdx, token: seqIdx, headOrNeuron: headIdx, isMlp: false, headReplica: 0 };
                const usedLoc = { layerWithIO: usedLayerIdx, token: usedSeqIdx, headOrNeuron: usedHeadIdx, isMlp: false, headReplica: 0 };
                const firstX = usedLocToXPos(usedLoc);
                const y = usedLocToYPos(usedLoc);
                const isHovered = isLocationHovered(hoveredPath, loc);
                heads.push(<g key={`${usedLayerIdx}-${usedSeqIdx}-${usedHeadIdx}`} transform={`translate(${firstX},${y})`}>
                    <rect x={-c.headWidth * 0.5} y={-c.headHeight * 0.5} height={c.headHeight} width={c.headWidth * head.paths.length} fill={colorNegPos(avg, c.huePos, c.hueNeg)} stroke="black" rx={2} strokeWidth={isHovered ? c.headHoveredStrokeWidth : 1}></rect>
                    {head.paths.map((path, pathIdx) => {
                        const pathToEnd = [...path, loc];
                        const values = getPathAttributionValues(stateToFakeTree(state, layerNamesWithIO), pathToEnd);
                        const value = values[values.length - 1];

                        const areQkvHovered = range(3).map(i => isLocationHovered(hoveredPath, { ...loc, qkv: i }, true));
                        return (<g transform={`translate(${c.headWidth * pathIdx},0)`} key={pathIdx} >
                            <rect x={-c.headWidth * 0.5} y={-c.headHeight * 0.5} height={c.headHeight} width={c.headWidth} fill={colorNegPos(value * 0.15, c.huePos, c.hueNeg)} stroke="#00000000" {...ohInner([...path, loc])} rx={2} ></rect>
                            <g transform={`translate(0,-2)`}>
                                <text fontWeight={areQkvHovered[0] ? "bold" : ""} transform={`translate(${c.qkvStride},0)`} textAnchor="middle" {...ohInner([...path, { ...loc, qkv: 0 }])}>q</text>
                                <text fontWeight={areQkvHovered[1] ? "bold" : ""} textAnchor="middle" transform={`translate(${-c.qkvStride},0)`} {...ohInner([...path, { ...loc, qkv: 1 }])}>k</text>
                                <text fontWeight={areQkvHovered[2] ? "bold" : ""} textAnchor="middle" {...ohInner([...path, { ...loc, qkv: 2 }])}>v</text>
                            </g>
                        </g>);
                    })}
                    <text fontWeight={isHovered ? "bold" : ""} transform={`translate(${c.headWidth * (head.paths.length - 1) / 2},${13})`} textAnchor="middle" style={{ pointerEvents: "none" }}>{headIdx}</text>
                </g>);
            });
        });
    });


    const neurons = [] as JSX.Element[];
    usedNeurons.forEach((layer, usedLayerIdx) => {
        if (usedLayers[usedLayerIdx] === 0 || usedLayers[usedLayerIdx] === layerNamesWithIO.length - 1) {
            return;
        }
        layer.forEach((seq, usedSeqIdx) => {
            seq.forEach((neuron, usedNeuronIdx) => {
                const layerIdx = usedLayers[usedLayerIdx];
                const seqIdx = usedTokens[usedSeqIdx];
                const headIdx = neuron.headOrNeuron;

                const loc = { layerWithIO: layerIdx, token: seqIdx, headOrNeuron: headIdx, isMlp: true, headReplica: 0 };
                const usedLoc = { layerWithIO: usedLayerIdx, token: usedSeqIdx, headOrNeuron: usedNeuronIdx, isMlp: true, headReplica: 0 };
                const isHovered = isLocationHovered(hoveredPath, loc);
                const firstX = usedLocToXPos(usedLoc);
                const y = usedLocToYPos(usedLoc);
                neurons.push(<g key={`${usedLayerIdx}-${usedSeqIdx}-${usedNeuronIdx}`} transform={`translate(${firstX},${y})`}>
                    <rect x={-c.neuronWidth * 0.5} y={-c.neuronHeight * 0.5} height={c.neuronHeight} width={c.neuronWidth * neuron.paths.length} fill="#00000000" stroke="black" rx={1}></rect>
                    {neuron.paths.map((path, pathIdx) => {
                        const pathToEnd = [...path, loc];
                        const values = getPathAttributionValues(stateToFakeTree(state, layerNamesWithIO), pathToEnd);
                        const value = values[values.length - 1];
                        return (<g key={pathIdx} transform={`translate(${c.neuronWidth * pathIdx},0)`}>
                            <rect x={-c.neuronWidth * 0.5} y={-c.neuronHeight * 0.5} height={c.neuronHeight} width={c.neuronWidth} fill={colorNegPos(value * 0.15, c.huePos, c.hueNeg)} stroke="#00000000" strokeWidth={isHovered ? c.headHoveredStrokeWidth : 1}></rect>
                            <text fontWeight={isHovered ? "bold" : ""} transform={`translate(${0},${3})`} textAnchor="middle" fontSize="8px" {...ohInner([...path, loc])}>{headIdx}</text>
                        </g>);
                    })}
                </g>);
            });
        });
    });


    const lines = [] as JSX.Element[];
    const lineLabels = [] as JSX.Element[];

    const renderLine = (usedFrom: AttribLocation, usedTo: AttribLocation, attribs: Attributions, baseFrom: AttribLocation, path: AttributionPath) => {
        const rawValue = indexAttributionsByLocation(attribs, baseFrom) as number;
        // console.log({rawValue,usedFrom,usedTo,attribs})
        const value = rawValue * stateSpec.lineWidthScale;
        let [x1, y1, x2, y2] = [
            usedLocToXPos(usedFrom),
            usedLocToYPos(usedFrom),
            usedLocToXPos(usedTo),
            usedLocToYPos(usedTo),
        ];
        if (baseFrom.layerWithIO !== 0) {
            if (usedFrom.isMlp) {
                y1 += c.neuronHeight / 2;
            } else {
                y1 += c.headHeight / 2;
            }
        }

        if (usedLayers[usedTo.layerWithIO] !== layerNamesWithIO.length - 1) {
            if (usedTo.isMlp) {
                y2 -= c.neuronHeight / 2;
            } else {
                y2 -= c.headHeight / 2;
            }
        }

        const pathToEnd = [...path, baseFrom];
        const thisHovered = hoveredPath && isOnePathPrefixOfOtherIgnoringQkvEnd(pathToEnd, hoveredPath);
        const yDist = y2 - y1;
        const pathString = `M${x1} ${y1}, ${c.useSplines ? `C${x1} ${y1 + yDist * c.splineSharpness}, ${x2} ${y2 - yDist * c.splineSharpness}, ${x2} ${y2}` : `L ${x2} ${y2}`}`;
        return [<g >
            <path style={{ pointerEvents: "none" }} d={pathString} stroke={colorNegPos(Math.sign(value), c.huePos, c.hueNeg)} strokeWidth={thisHovered ? c.lineWidthOnHover : c.lineWidthMax * Math.min(1, Math.abs(value))} fill="none" />
            {/* another thicker invisible line to make hovering easier */}
            <path d={pathString} {...ohInner(pathToEnd)} stroke={"#00000000"} strokeWidth={c.lineHoverMouseWidth} fill="none" />

        </g>,
        <g style={{ pointerEvents: "none" }}>
            <rect transform={`translate(${x1 - c.smallFontSize * 1.5},${y1 + 2})`} width={c.smallFontSize * 3} height={c.smallFontSize + c.wordMargin - 2} fill="white" />
            <text fontSize={`${c.smallFontSize}px`} transform={`translate(${x1},${y1 + c.smallFontSize + 1})`} textAnchor="middle">{rawValue.toFixed(3)}</text>
        </g>];
    };

    const walkTreeWritingSvgLines = (tree: AttributionTreeTensors, path: AttributionPath) => {
        const newPath = [...path, tree.idx];
        if (tree.idx.layerWithIO === layerNamesWithIO.length - 1) {
            newPath.pop();
        }
        for (let outgoing of tree.outgoingLines || [] as LocationUsedAndBase[]) {
            // console.log("rendering line");
            const [line, lineLabel] = renderLine(outgoing.used as AttribLocation, tree.usedIdx, tree.attribution, outgoing.base, newPath);
            lines.push(line);
            lineLabels.push(lineLabel);
        }
        tree.children.forEach(x => walkTreeWritingSvgLines(x, newPath));
    };
    walkTreeWritingSvgLines(fakeTree, []);

    console.log("to render bits", { lines, lineLabels, heads, neurons });
    return (
        <svg viewBox={`0 0 ${neededWidth} ${neededHeight}`} width={`${availableWidth}px`} height={`${availableHeight}px`} >
            <g transform={`translate(${c.maxWordWidth},${yTop})`}>

                <g transform={`translate(0,${6})`}>
                    {usedLayers.map((layer_idx, used_layer_idx) => layerPoses[used_layer_idx][0] !== -1 && (<text key={layer_idx} textAnchor="end" transform={`translate(0,${layerPoses[used_layer_idx][0]})`}>{`${used_layer_idx !== 0 && used_layer_idx !== usedLayers.length - 1 ? "attn " : ""}${layerNamesWithIO[layer_idx]}`}</text>))}
                    {usedLayers.map((layer_idx, used_layer_idx) => used_layer_idx !== 0 && layerPoses[used_layer_idx][1] !== -1 && (<text key={layer_idx} textAnchor="end" transform={`translate(0,${layerPoses[used_layer_idx][1]})`}>{`mlp ${layerNamesWithIO[layer_idx]}`}</text>))}
                </g>

                <g transform={`translate(0,${-c.fontSize - c.wordMargin})`} >
                    {usedTokens.map((seqIdx, used_seq_idx) => (<g key={seqIdx} ><text fontWeight={isLocationHovered(hoveredPath, { layerWithIO: 0, token: seqIdx, headOrNeuron: 0, isMlp: false }) ? "bold" : ""} textAnchor="middle" transform={`translate(${seqPoses[used_seq_idx]},0)`}>{tokenToInlineString(tokens[seqIdx])}</text>
                        {(usedHeads[0][used_seq_idx][0] || { paths: [0] }).paths.map((_, pathIdx) => {
                            return <text fontWeight={isLocationHovered(hoveredPath, { layerWithIO: 0, token: seqIdx, headOrNeuron: 0, isMlp: false }) ? "bold" : ""} transform={`translate(${usedLocToXPos({ layerWithIO: 0, token: used_seq_idx, headOrNeuron: 0, isMlp: false })},${c.fontSize + c.wordMargin})`} textAnchor="middle">{`${seqIdx}`}</text>;
                        })}</g>))}
                </g>

                <g className="around-lines">{lines}</g>
                <g className="around-heads">{heads}</g>
                <g className="around-neurons">{neurons}</g>
                <g className="around-line-labels">{lineLabels}</g>
                <g transform={`translate(0,${Math.max(...layerPoses[layerPoses.length - 2]) + 6})`}>
                    {usedTokens.map((seqIdx, used_seq_idx) => stateSpec.root && seqIdx === stateSpec.root.data.seqIdx && (
                        <text key={seqIdx} textAnchor="middle" transform={`translate(${seqPoses[used_seq_idx]},0)`} {...ohInner([])}>{attRootToString(stateSpec.root)}</text>))}
                </g>
            </g>
        </svg>);
}