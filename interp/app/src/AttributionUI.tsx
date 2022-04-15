import { useEffect, useRef, useState } from 'react';
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import { VeryNamedTensor, PickArgs, AttribLocation, AttributionPath, AttributionBackend, AttributionRoot, LogitRoot, AttributionTreeSpec, AttributionStateSpec, AttributionTreeTensors, AttributionTensors, QueriedLogits, Attributions, AttributionsManyOut, IMPLEMENTED_TOKEN_ROOTS } from "./proto";
import { range, getOverX, vntPick, deepPromiseAll, stateToFakeTree } from "./common";
import { colorNegPos, Select, ShowQueriedLogits, tokenToInlineString } from "./ui_common";
import ndarray from 'ndarray';
import { vocab } from "./tokens";
import { MultiHueText } from './MultiHueText';
import { getIndexMapperOfDiff } from "./diff";
import { AttributionUISvg, getTreePath, indexAttributionsByLocation } from "./AttributionUISvg";


const attributionsPickO = (attribs: AttributionsManyOut, i: number) => {
    return { embeds: vntPick(attribs.embeds, [i, null]), heads: vntPick(attribs.heads, [i, null, null, null]), mlps: vntPick(attribs.mlps, [i, null, null, null]), __notmanyout__: true };
};


// I need this debounce / netMouseEnters bullshit because mouseenter and mouseleave callbacks aren't always called in the order I want
let debounce = null as any;

export function AttributionUI(props: { options: any, urlState: AttributionStateSpec, setUrlState: (x: AttributionStateSpec) => void, attributionBackend: AttributionBackend; }) {

    const { options, urlState: stateSpec, setUrlState: _setStateSpec } = props;
    const Backend: AttributionBackend = props.attributionBackend as any;
    const { headNames, layerNames, tokens, neuronNames } = Backend;
    const aroundFigureDivRef = useRef(null as null | HTMLDivElement);

    const layerNamesWithIO = ["embeds", ...layerNames, "output"];

    const [hoveredPath, _setHoveredPath] = useState(null as null | AttributionPath);
    const [numOutstandingRequests, setNumOutstandingRequests] = useState(0);
    const defaultAttribRoot = { kind: "logprob", threshold: 0.1, data: { seqIdx: tokens.length - 2, tokString: tokens[tokens.length - 1], comparisonTokString: null } };
    const defaultSpec = { tree: [], root: defaultAttribRoot, lineWidthScale: 1, useIGAttn: false, useIGOutput: true, showNegative: true, useActivationVsMean: false, fuseNeurons: true, halfLinearActivation: false, threshold: 0.1, specificLogits: [], modelName: Backend.modelName, nonce: 0, toks: tokens } as AttributionStateSpec;
    const [pastStates, setPastStates] = useState([] as AttributionStateSpec[]);
    const [tensorStates, setTensorStates] = useState({ tree: [] as AttributionTreeTensors[], pathDirectionLogits: null as null | QueriedLogits, root: defaultAttribRoot } as AttributionTensors);
    const [viewportAspectRatio, setViewportAspectRatio] = useState(0.5);
    const [thresholdInProgress, setThresholdInProgress] = useState(stateSpec?.threshold ?? 0.1);

    const setStateSpec = (newStateSpec: AttributionStateSpec) => {
        _setStateSpec(newStateSpec);
        setPastStates([...pastStates, newStateSpec]);
    };


    const resetOutgoingLines = (states: AttributionTensors, passedStateSpec: AttributionStateSpec | null = null) => {
        if (passedStateSpec === null) {
            passedStateSpec = stateSpec;
        }
        // TODO: make ones to expanded nodes always show
        const recurse = (tree: AttributionTreeTensors) => {

            const embedsSparseTensor = getOverX(tree.attribution.embeds.tensor, tree.threshold, passedStateSpec?.showNegative);
            const embedsOutgoing = embedsSparseTensor.idxs.map((x, i) => ({ base: { layerWithIO: 0, headOrNeuron: 0, isMlp: false, token: x[0] } }));

            const headsSparseTensor = getOverX(tree.attribution.heads.tensor, tree.threshold, passedStateSpec?.showNegative);
            const headsOutgoing = headsSparseTensor.idxs.map((x, i) => ({ base: { layerWithIO: x[0] + 1, headOrNeuron: x[1], isMlp: false, token: x[2] } }));

            const mlpsSparseTensor = getOverX(tree.attribution.mlps.tensor, tree.threshold, passedStateSpec?.showNegative);
            const mlpsOutgoing = mlpsSparseTensor.idxs.map((x, i) => ({ base: { layerWithIO: x[0] + 1, headOrNeuron: x[1], isMlp: true, token: x[2] } }));

            tree.children = tree.children.map(recurse);
            return { ...tree, outgoingLines: [...embedsOutgoing, ...headsOutgoing, ...mlpsOutgoing] };
        };
        const tree = recurse(stateToFakeTree(states, layerNamesWithIO)); // qkv set to 2 because that's visually in the middle?
        return { ...states, root: { ...states.root, outgoingLines: tree.outgoingLines }, tree: tree.children } as AttributionTensors;
    };

    const resetFromStateSpec = (stateSpec: AttributionStateSpec) => {
        const attributionsTableBecausePathsAreByQKVButRequestsArent = {} as { [x: string]: Promise<AttributionsManyOut>; };
        console.log("FUSE NEURONS", stateSpec.fuseNeurons);
        const recurse: (tree: AttributionTreeSpec, path: AttributionPath) => any = (tree: AttributionTreeSpec, path: AttributionPath) => {
            path = [...path, tree.idx];
            const qPath = [...path.slice(0, path.length - 1), { ...path[path.length - 1] }];
            const qkv = qPath[qPath.length - 1].qkv;
            delete qPath[qPath.length - 1].qkv;
            let promise = attributionsTableBecausePathsAreByQKVButRequestsArent[JSON.stringify(qPath)];
            if (promise === undefined) {
                attributionsTableBecausePathsAreByQKVButRequestsArent[JSON.stringify(qPath)] = Backend._expandTreeNode(qPath, stateSpec.useIGAttn, stateSpec.fuseNeurons, stateSpec.halfLinearActivation);
                promise = attributionsTableBecausePathsAreByQKVButRequestsArent[JSON.stringify(qPath)];
            }
            let promiseHere: Promise<Attributions> = qkv !== undefined ?
                promise.then(x => attributionsPickO(x, qkv)) :
                promise.then(x => attributionsPickO(x, 0));

            return { ...tree, children: tree.children.map((x) => recurse(x, path)), attribution: promiseHere };
        };
        const treeOfPromises = { startTree: Backend._startTree(stateSpec.root, stateSpec.useIGOutput, stateSpec.useActivationVsMean, stateSpec.fuseNeurons).then(x => attributionsPickO(x, 0)), children: stateSpec.tree.map(t => recurse(t, [])) };
        deepPromiseAll(treeOfPromises).then((tree: any) => {
            const { startTree, children } = tree;
            const stateTensors = { ...stateSpec, tree: children, pathDirectionLogits: null, root: { ...stateSpec.root, attribution: startTree } } as AttributionTensors;
            setTensorStates(resetOutgoingLines(stateTensors, stateSpec));
            setStateSpec(stateSpec);
        });
    };

    if (!stateSpec) {
        resetFromStateSpec(defaultSpec);
        return (<p>Loading</p>);
    }
    if (!tensorStates.root.attribution) {
        resetFromStateSpec(stateSpec);
        return (<p>Loading</p>);
    }
    // if (stateSpec.nonce!==tensorStates.nonce){
    //     return (<p>Nonces don't match</p>)
    // }

    // if backend changes, update seq indexes in tree based on diff of tokens in state and reset from state
    {
        const newToks = Backend.tokens;
        const oldToks = stateSpec.toks;
        if (JSON.stringify(newToks) !== JSON.stringify(oldToks) && Backend.modelName === stateSpec.modelName) {

            const idxMapper = getIndexMapperOfDiff(oldToks, newToks);
            const newStateSpec = { ...stateSpec,toks:newToks };
            const newRootIdx = idxMapper(stateSpec.root.data.seqIdx);
            if (newRootIdx === null) {
                resetFromStateSpec(defaultSpec);
                return <p>Loading</p>;
            } else {
                newStateSpec.root = { ...newStateSpec.root, data: { ...newStateSpec.root.data, seqIdx: newRootIdx } };
                const recurse: (tree: AttributionTreeSpec) => AttributionTreeSpec | null = (tree: AttributionTreeSpec) => {
                    const newSeqIdx = idxMapper(tree.idx.token);
                    if (newSeqIdx === null) {
                        return null;
                    }
                    const newIdx = { ...tree.idx, token: newSeqIdx };
                    return { ...tree, idx: newIdx, children: tree.children.map(recurse).filter(x => x !== null) } as AttributionTreeSpec;
                };
                newStateSpec.tree = newStateSpec.tree.map(x => recurse(x)).filter(x => x !== null) as AttributionTreeSpec[];
                console.log({ newStateSpec });
                resetFromStateSpec(newStateSpec);
                return <p>Loading</p>;
            }
        }
    }

    const setTreeRoot = (root: AttributionRoot) => {
        console.log("SET TREE ROOT");
        if (IMPLEMENTED_TOKEN_ROOTS[root.kind] && IMPLEMENTED_TOKEN_ROOTS[stateSpec.root.kind] && root.data.seqIdx === stateSpec.root.data.seqIdx) {
            resetFromStateSpec({ ...stateSpec, root: root });
            return;
        }
        setNumOutstandingRequests(x => x + 1);
        Backend._startTree(root, stateSpec.useIGOutput, stateSpec.useActivationVsMean, stateSpec.fuseNeurons).then((newAttributions: AttributionsManyOut) => {
            console.log("starttree attribs", newAttributions);
            const newNonce = stateSpec.nonce + 1;
            setStateSpec({ ...stateSpec, tree: [], root, nonce: newNonce });
            const s = {
                tree: [],
                pathDirectionLogits: null,
                root: { ...root, attribution: attributionsPickO(newAttributions, 0) },
                nonce: newNonce
            } as AttributionTensors;
            setTensorStates(resetOutgoingLines(s));
            setNumOutstandingRequests(x => x - 1);
        });
    };


    const expandFromHead = (treePath: AttributionPath) => {
        if (getTreePath(stateSpec, treePath, layerNamesWithIO)) {
            console.log("already expanded that");
            resetFromStateSpec(stateSpec);
            return;
        }
        setNumOutstandingRequests(x => x + 1);
        const pathLeaf = treePath[treePath.length - 1];
        const threshold = stateSpec.threshold;
        console.log("FUSE NEURONS", stateSpec.fuseNeurons);
        Backend._expandTreeNode(treePath, stateSpec.useIGAttn, stateSpec.fuseNeurons, stateSpec.halfLinearActivation).then((attribsManyOut: AttributionsManyOut) => {
            const newStateTensors = { ...tensorStates };
            const newStateSpec = { ...stateSpec };
            console.log("EXPANDING NODE", { attribsManyOut, tensorStates, stateSpec });
            if (pathLeaf.isMlp) {
                getTreePath(newStateSpec, treePath.slice(0, treePath.length - 1), layerNamesWithIO).children.push(...range(3).map(qkv => ({ idx: { ...pathLeaf, qkv }, children: [], threshold }) as any));
                getTreePath(newStateTensors, treePath.slice(0, treePath.length - 1), layerNamesWithIO).children.push(...range(3).map(qkv => ({ idx: { ...pathLeaf, qkv }, children: [], attribution: attributionsPickO(attribsManyOut, qkv), threshold, usedIdx: { ...pathLeaf, qkv } } as AttributionTreeTensors)));
            } else {
                getTreePath(newStateSpec, treePath.slice(0, treePath.length - 1), layerNamesWithIO).children.push(...range(3).map(qkv => ({ idx: { ...pathLeaf, qkv }, children: [], threshold } as any)));
                getTreePath(newStateTensors, treePath.slice(0, treePath.length - 1), layerNamesWithIO).children.push(...range(3).map(qkv => ({ idx: { ...pathLeaf, qkv }, children: [], attribution: attributionsPickO(attribsManyOut, qkv), threshold, usedIdx: { ...pathLeaf, qkv } } as AttributionTreeTensors)));
            }

            setTensorStates(resetOutgoingLines(newStateTensors));
            setStateSpec(newStateSpec);
            setNumOutstandingRequests(x => x - 1);
        });
    };

    const setHoveredPath = (path: AttributionPath | null) => {
        if (path) { // backend only does length 2+ paths rn
            if (Backend._sparseLogitsForSpecificPath) {
                console.log("querying logits for path", path);
                const specificLogits = [...stateSpec.specificLogits];
                if (IMPLEMENTED_TOKEN_ROOTS[stateSpec.root.kind]) {
                    const theData = stateSpec.root.data as { tokString: string, comparisonTokString: string; };
                    specificLogits.push(theData.tokString);
                    if (theData.comparisonTokString) {
                        specificLogits.push(theData.comparisonTokString);
                    }
                }
                Backend._sparseLogitsForSpecificPath(path, stateSpec.useIGOutput ? "ig" : "none", stateSpec.useIGAttn ? "ig" : "none", specificLogits, stateSpec.fuseNeurons).then((queriedLogits) => {
                    setTensorStates(tensorStates => ({ ...tensorStates, pathDirectionLogits: queriedLogits }));
                });
            } else {
                console.warn("backend doesn't have _getTopkLogits");
            }
        } else {
            setTensorStates(tensorStates => ({ ...tensorStates, pathDirectionLogits: null }));
        }
        _setHoveredPath(path);
    };

    const removeFocus = () => {
        setHoveredPath(null);
    };

    const onMouseLeave = () => {
        if (debounce !== null) {
            clearTimeout(debounce);
        }
        debounce = setTimeout(() => removeFocus(), 100);
    };

    const onMouseEnter = (path: AttributionPath) => {
        if (debounce !== null) {
            clearTimeout(debounce);
            debounce = null;
        }
        console.log("SETTING HOVERED PATH", path);
        setHoveredPath(path);
    };
    const isTokenShown = tokens.map(x => false);
    const recurseForAllSeqIdxs = (tree: AttributionTreeTensors) => {
        tree.outgoingLines?.forEach(x => { isTokenShown[x.base.token] = true; });
        tree.children.forEach(recurseForAllSeqIdxs);
    };
    recurseForAllSeqIdxs(stateToFakeTree(tensorStates, layerNamesWithIO));
    const hoverReleventSeqIdxs = tokens.map((w, i) => hoveredPath === null ? false : hoveredPath.some(x => x.token === i));
    // console.log({ hoverReleventSeqIdxs, tokens, hoveredPath });
    const inputSums = false || stateSpec.root.attribution?.embeds.tensor;
    const usedTokensStridedArr = [...tokens.map((_, i) => stateSpec.root.data.seqIdx === i ? 1 : 0), ...isTokenShown, ...hoverReleventSeqIdxs]; //@TODO show used tokens even though that's only computed in inner component?

    const usedTokensVNT = { dim_names: ["UNK", "used_tokens"], dim_types: ["UNK", "seq"], dim_idx_names: [["loss_on", "shown_words", "hover"], tokens], tensor: ndarray(usedTokensStridedArr, [3, tokens.length]), units: "shown below", title: "shown tokens" };

    const ohInner = (path: AttributionPath) => {
        return {
            onMouseLeave: onMouseLeave,
            onMouseEnter: () => {
                onMouseEnter(path);
            },
            onClick: () => {
                if (path.length > 0) {
                    expandFromHead(path);
                }
            }
        };
    };


    const undo = () => {
        if (pastStates.length > 0) {
            const lastState = pastStates[pastStates.length - 1];
            setPastStates(pastStates.slice(0, pastStates.length - 1));
            resetFromStateSpec(lastState);
        }
    };
    const availableWidth = aroundFigureDivRef.current ? aroundFigureDivRef.current.getBoundingClientRect().width : 600;

    return (<div className="box border_green" style={{ marginTop: "12px" }}>
        <div className="section_title" >
            Path Attribution  </div>


        <div>
            {numOutstandingRequests > 0 ? <span style={{ backgroundColor: "yellow" }}>Waiting for Server</span> : <span>​</span>}
        </div>
        <div className="chart" style={{ display: "table", margin: "auto" }}>
            <MultiHueText vnt={usedTokensVNT as VeryNamedTensor} hover={[null, null]} setFocus={(dims: PickArgs) => {// TODO make this work for hovering anything
                if (dims[1] !== null) {
                    setTreeRoot({ kind: IMPLEMENTED_TOKEN_ROOTS[stateSpec.root.kind] ? stateSpec.root.kind : "logit", data: { seqIdx: dims[1], tokString: Backend.tokens[Math.min(Backend.tokens.length - 1, dims[1] + 1)], comparisonTokString: null }, threshold: stateSpec.threshold });
                }
            }} setHover={(dims: PickArgs) => {
                if (dims[1] === tensorStates.root.data.seqIdx) { // TODO make this work for hovering anything
                    onMouseEnter([]);
                }
            }} highlight={[null, null]}
            />
        </div>
        <div className="attribution_chart">
            <div ref={aroundFigureDivRef} tabIndex={0} className="attributionuiaroundsvg" onKeyDown={(e) => {
                console.log("KEYDOWN", e);
                if (e.altKey) {
                    const MAC_ALT_NUMBER_KEYS = `o¡™£¢∞§¶•ª`;
                    const keyInt = parseInt(e.key) || MAC_ALT_NUMBER_KEYS.indexOf(e.key);
                    if (keyInt && keyInt !== -1) {
                        resetFromStateSpec({ ...stateSpec, threshold: parseFloat(Math.exp(1.1 - keyInt).toPrecision(3)) }); // 3 to 0.0004, geometrically by intervals of e
                    }
                } else if (e.ctrlKey && e.key === "z") {
                    console.log("Undoing", pastStates);
                    undo();
                    e.preventDefault();
                }
            }} onClick={(e) => ((e.target as HTMLElement).closest(".attributionuiaroundsvg") as any).focus()} style={{ outline: "none" }}>
                <TransformWrapper limitToBounds={false} minScale={0.2} wheel={{ step: 0.1, activationKeys: ["Shift", "Control", "ctrl", "shft", "Alt", "alt"] }} doubleClick={{ disabled: true }} >
                    <TransformComponent>
                        <AttributionUISvg state={tensorStates} ohInner={ohInner} layerNamesWithIO={layerNamesWithIO} tokens={tokens} hoveredPath={hoveredPath} stateSpec={stateSpec} availableWidth={availableWidth} availableHeight={availableWidth * viewportAspectRatio} />
                    </TransformComponent>
                </TransformWrapper>
            </div>
        </div>
        <div style={{ display: "flex", flexDirection: "row", alignItems: "flex-start", justifyContent: "space-between" }}>
            <div className="box border_green">
                {/* UNDO is broken right now, am disabling <div>
                    <button className="box border_green" onClick={undo} >Undo</button>
                </div> */}
                {stateSpec.root && (IMPLEMENTED_TOKEN_ROOTS[stateSpec.root.kind]) && <>
                    <label style={{ margin: "0" }}>Target Token</label>
                    <Select sortedWords={vocab} value={(stateSpec.root.data as LogitRoot).tokString} onChange={(s: string) => setTreeRoot({ kind: stateSpec.root.kind, data: { ...stateSpec.root.data, tokString: s }, threshold: stateSpec.threshold })} defaultTopWords={tokens} placeholder={"Target Token"}></Select>
                    <br />
                    <label style={{ margin: "0" }}>Comparison Token</label>
                    <Select sortedWords={vocab} value={(stateSpec.root.data as LogitRoot).comparisonTokString} onChange={(s: string) => setTreeRoot({ kind: stateSpec.root.kind, data: { ...stateSpec.root.data, comparisonTokString: s }, threshold: stateSpec.threshold })} defaultTopWords={tokens} showClearButton={true} placeholder={"Comparison Token"}></Select></>}
            </div>

            <div className="box border_green">
                <div className="box_title"> Logits at this position: </div>

                {tensorStates.pathDirectionLogits && <ShowQueriedLogits ql={tensorStates.pathDirectionLogits} />}

                <label style={{ margin: "0" }}>Add Specific Token</label>
                <div style={{ display: "flex", flexDirection: "row", flexWrap: "wrap" }}>
                    <Select sortedWords={vocab} value={null} onChange={(s: string) => setStateSpec({ ...stateSpec, specificLogits: [...stateSpec.specificLogits, s] })} defaultTopWords={[]} placeholder={"Add Token"}></Select>


                    <div style={{ margin: "0 10px", whiteSpace: "nowrap" }}>
                        {stateSpec.specificLogits.map((x, i) => (
                            <div key={i} style={{ padding: "1px 3px", margin: "4px 2px", border: "1px solid black", cursor: "pointer", borderRadius: "3px" }} onClick={() => {
                                const newSpecificLogits = [...stateSpec.specificLogits];
                                newSpecificLogits.splice(i, 1);
                                setStateSpec({ ...stateSpec, specificLogits: newSpecificLogits });
                            }}>
                                ✖&nbsp;{tokenToInlineString(x)}
                            </div>))}
                    </div>
                </div>

            </div>
            <div className="box border_green" title="set threshold with the keyboard shortcut Alt+Number">
                <div className="box_title"> Display options: </div>
                <table>
                    <tr> 
                        <td>Line visibility threshold &nbsp;</td>
                        <td><input style={{ width: "60px", border: "1px solid black", fontSize: "inherit" }} type="number" step="0.1" value={stateSpec.threshold} onChange={(e) => setStateSpec({ ...stateSpec, threshold: parseFloat(e.target.value) })} onKeyDown={(e)=>{
                            if (e.key==="Enter"){
                                const newStateSpec = {...stateSpec}
                                const fakeTreeSpec = stateToFakeTree(stateSpec,layerNamesWithIO)
                                const fakeTreeTensors = stateToFakeTree(tensorStates,layerNamesWithIO)
                                const recurse: (spec:AttributionTreeSpec,tensors:AttributionTreeTensors)=>AttributionTreeSpec= (spec:AttributionTreeSpec,tensors:AttributionTreeTensors)=>{
                                    return {...spec, threshold:stateSpec.threshold, children:range(spec.children.length).filter(i=>indexAttributionsByLocation(tensors.attribution, spec.children[i].idx)>=stateSpec.threshold).map(i=>recurse(spec.children[i],tensors.children[i]))} as AttributionTreeSpec
                                }
                                const newFakeTree = recurse(fakeTreeSpec,fakeTreeTensors)
                                newStateSpec.root = {...newStateSpec.root, threshold:newFakeTree.threshold}
                                newStateSpec.tree = newFakeTree.children
                                resetFromStateSpec(newStateSpec)
                            }
                        }}/></td>
                    </tr>
                    <tr>
                        <td>Line width scale </td>
                        <td><input style={{ width: "60px", border: "1px solid black", fontSize: "inherit" }} type="number" value={stateSpec.lineWidthScale} onChange={(e) => setStateSpec({ ...stateSpec, lineWidthScale: parseFloat(e.target.value) })} /></td>
                    </tr>
                    <tr>
                        <td>Aspect ratio</td>
                        <td><input style={{ width: "60px", border: "1px solid black", fontSize: "inherit" }} type="number" value={viewportAspectRatio} onChange={(e) => setViewportAspectRatio(parseFloat(e.target.value))}></input></td>
                    </tr>

                    <tr>
                        <td> Integrate gradients for attn probs &nbsp; </td>
                        <td>
                            <label className="switch">
                                <input type="checkbox" checked={stateSpec.useIGAttn} onClick={() => resetFromStateSpec({ ...stateSpec, useIGAttn: !stateSpec.useIGAttn })} />
                                <span className="slider round"></span>
                            </label>
                        </td>
                    </tr>

                    <tr>
                        <td> Integrate gradients for loss log probs &nbsp; </td>
                        <td>
                            <label className="switch">
                                <input type="checkbox" checked={stateSpec.useIGOutput} onClick={() => resetFromStateSpec({ ...stateSpec, useIGOutput: !stateSpec.useIGOutput })} />
                                <span className="slider round"></span>
                            </label>
                        </td>
                    </tr>

                    <tr>
                        <td> Show negative attribution &nbsp; </td>
                        <td>
                            <label className="switch">
                                <input type="checkbox" checked={stateSpec.showNegative} onClick={() => resetFromStateSpec({ ...stateSpec, showNegative: !stateSpec.showNegative })} />
                                <span className="slider round"></span>
                            </label>
                        </td>
                    </tr>

                    <tr>
                        <td> Subtract Dataset Mean from Outputs &nbsp; </td>
                        <td>
                            <label className="switch">
                                <input type="checkbox" checked={stateSpec.useActivationVsMean} onClick={() => resetFromStateSpec({ ...stateSpec, useActivationVsMean: !stateSpec.useActivationVsMean })} />
                                <span className="slider round"></span>
                            </label>
                        </td>
                    </tr>
                    {/* don't allow unfused neurons in "low memory" mode */}
                    {!process.env.REACT_APP_LOW_MEMORY && 
                    <tr>
                        <td> Fuse Neurons &nbsp; </td>
                        <td>
                            <label className="switch">
                                <input type="checkbox" checked={stateSpec.fuseNeurons} onClick={() => resetFromStateSpec({ ...stateSpec, fuseNeurons: !stateSpec.fuseNeurons })} />
                                <span className="slider round"></span>
                            </label>
                        </td>
                    </tr>}
                    <tr>
                        <td> Leaky Activations &nbsp; </td>
                        <td>
                            <label className="switch">
                                <input type="checkbox" checked={stateSpec.halfLinearActivation} onClick={() => resetFromStateSpec({ ...stateSpec, halfLinearActivation: !stateSpec.halfLinearActivation })} />
                                <span className="slider round"></span>
                            </label>
                        </td>
                    </tr>

                    {stateSpec.root && (IMPLEMENTED_TOKEN_ROOTS[stateSpec.root.kind]) &&
                        (<>
                            <tr><td> <span style={{ fontWeight: stateSpec.root.kind === "logit" ? "bold" : "" }}>Logits?</span></td>
                                <td>
                                    <label className="switch">
                                        <input type="checkbox" checked={stateSpec.root.kind === "logit"} onClick={() => setTreeRoot({ ...stateSpec.root, kind: "logit" })} />
                                        <span className="slider round"></span>
                                    </label>
                                </td></tr>
                            <tr><td> <span style={{ fontWeight: stateSpec.root.kind === "logprob" ? "bold" : "" }}>Logprobs?</span></td>
                                <td>
                                    <label className="switch">
                                        <input type="checkbox" checked={stateSpec.root.kind === "logprob"} onClick={() => setTreeRoot({ ...stateSpec.root, kind: "logprob" })} />
                                        <span className="slider round"></span>
                                    </label>
                                </td></tr>
                            <tr><td> <span style={{ fontWeight: stateSpec.root.kind === "prob" ? "bold" : "" }}>Probs?</span></td>
                                <td>
                                    <label className="switch">
                                        <input type="checkbox" checked={stateSpec.root.kind === "prob"} onClick={() => setTreeRoot({ ...stateSpec.root, kind: "prob" })} />
                                        <span className="slider round"></span>
                                    </label>
                                </td></tr>
                        </>)}
                </table>
            </div>
        </div>
        <div>Reminder: Does that token need a space?</div>
    </div>
    );
};

