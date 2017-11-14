let seed: int = [%bs.raw "parseInt(Math.random() * Number.MAX_SAFE_INTEGER)"];
Random.init(seed);

type transferFunc = {
  func: (float) => float,
  deriv: (float) => float
};

type input = Perceptron(perceptron) | Input(float)
and outputs = Perceptrons(array(perceptron)) | Output(float)
and perceptron = {
  mutable inputNodes: array(input),
  mutable outputNodes: outputs,
  mutable weights: array(float),
  mutable output: float,
  mutable invalid: bool,
  mutable dEdN: float,
  mutable transfer: transferFunc,
};

type networkInputs = array(input);

let makeWeights = fun(length: int) => {
  Array.map(
    fun(_) => Random.float(1.0),
    Array.make(length, 0.0));
};

let relu = {
  func: fun(inp: float) => {
    log(1.0 +. exp(inp));
  },
  deriv: fun(inp: float) => {
    1.0 /. (1.0 +. exp(-. inp));
  }
};

let linear ={
  func: fun(inp: float) => inp,
  deriv: fun(_) => 1.0
};

let tanhAct = {
  func: fun(inp: float) => tanh(inp),
  deriv: fun(inp: float) => 1.0 -. (tanh(inp) ** 2.0)
};

let sum = fun(arr: array(float)) => {
  Array.fold_right(
    (x, y) => x +. y,
    arr,
    0.0
  );
};

let rec calcOutput = fun(perc: perceptron) => {
  switch perc.invalid {
    /* perceptron value needs to be recalculated */
    | true => {
      /* multiply inputNode values by weights and pass through transfer function */
      perc.output = perc.transfer.func(Array.fold_right(
        (x, y) => x +. y,
        Array.mapi(
          (ind, inputNode) => {
            switch inputNode {
              | Perceptron(thisPerc) => calcOutput(thisPerc) *. perc.weights[ind]
              | Input(inp) => inp *. perc.weights[ind];
            }
          },
          perc.inputNodes
        ),
        0.0
      ));
      /* Revalidate output */
      perc.invalid = false;
      /* If this is an output node, update the outputNode value */
      ignore(switch perc.outputNodes {
        | Output(_) => perc.outputNodes = Output(perc.output)
        | Perceptrons(_) => ();
      });
    
      perc.output;
    }
    | false => perc.output
  }
};

let makePerceptron = fun(inputs: array(input), transfer: transferFunc) => {
  let perc = {
    /* Add an input of 1.0 for bias */
    inputNodes: Array.append(inputs, [|Input(1.0)|]),
    outputNodes: Output(0.0),
    /* An extra weight for the bias */
    weights: makeWeights(Array.length(inputs) + 1),
    output: 0.0,
    dEdN: 0.0,
    invalid: true,
    transfer: transfer
  };

  /* Update any input nodes which are perceptrons to have the new node as an output */
  ignore(Array.iter(
    (thisPerc) => {
      switch thisPerc {
        | Perceptron(thisThisPerc) => {
          thisThisPerc.outputNodes = switch thisThisPerc.outputNodes {
            | Perceptrons(percs) => Perceptrons(Array.append(percs, [|perc|]))
            | Output(_) => Perceptrons([|perc|])
          }
        }
        | Input(_) => ();
      }
    },
    inputs
  ));

  perc;
};

type network = array(array(perceptron));

let rec recursivelyMakeNetwork = fun(transfer: transferFunc, shape: list(int), net: option(network)) => {
  let inputs = switch net {
    | None => {
      Array.make(List.hd(shape), Input(0.0));
    }
    | Some(net) => {
      Array.map(
        (perc) => Perceptron(perc),
        net[Array.length(net) - 1]
      );
    }
  };

  let existing = switch net {
    | None => [||]
    | Some(net) => net
  };

  switch shape {
    | [] => {
      existing;
    }
    | [layerSize, ...layers] => {
      recursivelyMakeNetwork(
        transfer,
        layers,
        Some(Array.append(
          existing,
          [|Array.map(
            (_) => makePerceptron(inputs, transfer),
            Array.make(layerSize, 0)
          )|]
        ))
      )
    }
  }
};

let makeNetwork = fun(tranfer: transferFunc, shape: list(int)) => {
  recursivelyMakeNetwork(tranfer, shape, None);
};

let calcOutput = fun(net: network) => {
  let outputLayer = net[Array.length(net) - 1];
  Array.map(
    (perc) => calcOutput(perc),
    outputLayer
  );
};

let setInputs = fun(net: network, inputs: networkInputs) => {
  let inputLayer = net[0];
  ignore(Array.iter(
    (perc) => {
      perc.inputNodes = inputs;
      /* Check if all inputs are floats */
      /* If so, reduce the node weight array length to not include bias */
      /* Otherwise warn that connecting more perceptrons this way is weird */
      if (Array.fold_right(
        (input, allFloats) => switch input {
          | Perceptron(_) => false;
          | Input(_) => allFloats
        },
        inputs,
        true
      )) {
        ignore(Array.map(
          (perc) => perc.weights = Array.sub(perc.weights, 0, Array.length(inputs)),
          inputLayer
        ));
      } else {
        Js.log("WARNING - you are connecting perceptrons rather than values as network inputs");
      };
    },
    inputLayer
  ));
  ignore(Array.iter(
    (layer) => Array.iter(
      (perc) => perc.invalid = true,
      layer
    ),
    net
  ));

  net;
};

exception WrongNodeType(string);

let calcError = fun(net: network, expectedOutput: array(outputs)) => {
  let outputs = calcOutput(net);
  0.5 *. sum(Array.mapi(
    (ind, thisOutput) => {
      switch thisOutput {
        | Output(x) => (x -. outputs[ind]) ** 2.0
        | Perceptrons(_) => raise(WrongNodeType("Perceptron"));
      }
    },
    expectedOutput
  ));
};

let logOutput= fun(net: network) {
  Js.log("Inputs");
  ignore(Array.iter(
    (inputNode) => switch inputNode {
      | Perceptron(_) => Js.log("Perceptron")
      | Input(inp) => Js.log(inp)
    },
    net[0][0].inputNodes
  ));
  Array.mapi(
    (ind, layer) => {
      Js.log("Layer " ++ string_of_int(ind));
      Array.map(
        (perc) => Js.log(perc.output),
        layer
      )
    },
    net
  );
};

let logWeights = fun(net: network) {
  ignore(Array.mapi(
    (ind, layer) => {
      Js.log("Layer " ++ string_of_int(ind));
      Array.map(
        (perc) => Js.log(perc.weights),
        layer
      )
    },
    net
  ));
};

type linearObj = Vector(array(float)) | Matrix(array(linearObj));
exception DimensionMismatch(string);

let leftPad = fun(s: string, indent: int) {
  let retString = ref(s);
  for (_ in 0 to indent) {
    retString := " " ++ retString^;
  };
  retString^
};

let rec logLinearObj = fun(obj: linearObj, indent: int) {
  switch obj {
    | Vector(v) => {
      Js.log(leftPad(Array.fold_left(
        (output, x) => output ++ " " ++ string_of_float(x),
        leftPad("", indent),
        v
      ), indent));
    }
    | Matrix(m) => {
      Array.iteri(
        (ind, innerObj) => {
          Js.log(leftPad(string_of_int(ind), indent));
          logLinearObj(innerObj, indent + 2);
        },
        m
      );
    }
  }
};

let rec linearAdd = fun(a: linearObj, b: linearObj): linearObj {
  switch a {
    | Vector(u) => {
      switch b {
        | Vector(v) => {
          Vector(Array.mapi(
            (ind, entry) => entry +. u[ind],
            v
          ));
        }
        | Matrix(_) => {
          raise(DimensionMismatch("Objects do not have the same dimensions"));
        }
      }
    }
    | Matrix(m) => {
      switch b {
        | Vector(_) => {
          raise(DimensionMismatch("Objects do not have the same dimensions"));
        }
        | Matrix(n) => {
          Matrix(Array.mapi(
            (ind, entry) => linearAdd(entry, n[ind]),
            m
          ));
        }
      }
    }
  };
};

let rec linearClip = fun(clipLimit: float, obj: linearObj) {
  switch obj {
    | Vector(v) => {
      Vector(Array.map(
        (x) => copysign(min(clipLimit, abs_float(x)), x),
        v
      ))
    }
    | Matrix(m) => {
      Matrix(Array.map(
        linearClip(clipLimit),
        m
      ));
    }
  }
};


let sumDerivs = fun(derivs: list(linearObj)) {
  List.fold_right(
    (thisError, errorSum) => linearAdd(thisError, errorSum),
    List.tl(derivs),
    List.hd(derivs)
  )
};

let errorDerivs = fun(net: network, expectedOutput: array(outputs)) {
  let reversed = Array.of_list(List.rev(Array.to_list(net)));
  Matrix(Array.of_list(List.rev(Array.to_list(Array.map(
    (layer) => {
      Matrix(Array.mapi(
        (ind, perc) => {
          let dEdO = switch perc.outputNodes {
            | Output(output) => {
              let thisExpectedOutput = expectedOutput[ind];
              switch thisExpectedOutput {
                | Output(floatOutput) => output -. floatOutput
                | Perceptrons(_) => raise(WrongNodeType("Perceptron"))
              };
            }
            | Perceptrons(perceptrons) => {
              sum(Array.mapi(
                (innerInd, innerPerc) => {
                  innerPerc.dEdN *. innerPerc.weights[innerInd]
                },
                perceptrons
              ));
            }
          };
          let dOdN = perc.transfer.deriv(perc.output);
          perc.dEdN = dEdO *. dOdN;
          Vector(Array.map(
            (input) => {
              perc.dEdN *. switch input {
              | Perceptron(innerPerc) => innerPerc.output
              | Input(float) => float
            };
          },
          perc.inputNodes
        ));
      },
      layer
      ));
    },
    reversed
  )))));
};

let updateWeights = fun(net: network, errorDerivs: linearObj, alpha: float) {
  switch errorDerivs {
    | Matrix(layers) => {
      ignore(Array.mapi(
        (indA, layer) => {
          switch layer {
            | Matrix(percEntries) => {
              ignore(Array.mapi(
                (indB, percEntry) => {
                  switch percEntry {
                    | Vector(weightErrors) => {
                      ignore(Array.mapi(
                        (indC, weightError) => {
                          let perc = net[indA][indB];
                          let currentWeight = perc.weights[indC];
                          Array.set(perc.weights, indC, currentWeight -. (alpha *. weightError));
                          switch perc.outputNodes {
                            | Perceptrons(outputPercs) => {
                              ignore(Array.map(
                                (outputPerc) => outputPerc.invalid = true,
                                outputPercs
                              ))
                            }
                            | Output(_) => ()
                          }
                        },
                        weightErrors
                      ));
                    }
                    | _ => raise(DimensionMismatch("Object does not have the right number of dimensions"));
                  }
                },
                percEntries
              ))
            }
            | _ => raise(DimensionMismatch("Object does not have the right number of dimensions"));
          }
        },
        layers
      ))
    }
    | _ => raise(DimensionMismatch("Object does not have the right number of dimensions"));
  }
};

type trainingData = list((array(input), array(outputs)));

let sample = fun(min: float, max: float) {
  Random.float(max -. min) +. min;
};

let makeSingleValFunctionData = fun(f: (float) => float, min: float, max: float, number: int): trainingData {
  let examples = ref([]);
  for (_ in 0 to number) {
    let input = sample(min, max);
    examples := [([| Input(input) |], [| Output(f(input)) |]), ...examples^];
  };
  examples^;
};

let trainExample = fun(net: network, errorData: option((linearObj, float)), example: (array(input), array(outputs))) {
  let (inputs, outputs) = example;
  ignore(setInputs(net, inputs));
  ignore(calcOutput(net));
  let error = calcError(net, outputs);
  let derivs = errorDerivs(net, outputs);
  switch errorData {
    | None => (derivs, error)
    | Some(errorDataSoFar) => {
      let (weightErrorsSoFar, errorSum) = errorDataSoFar;
      (linearAdd(weightErrorsSoFar, derivs), error +. errorSum)
    }
  };
};

type trainingConfig = {
  shuffle: bool,
  gradientClip: float,
  maxEpochs: int,
  targetError: float,
  alpha: float,
  batchSize: int
};

let shuffleArray = fun(a: array('x)) {
  let b = Array.copy(a);
  Array.fast_sort(
    (_, _) => Random.int(2) - 1,
    b
  );
  b;
};

let trainMiniBatch = fun(config: trainingConfig, net: network, data: trainingData) {
  let errorData = List.fold_left(
    (errorSoFar, example) => Some(trainExample(net, errorSoFar, example)),
    None,
    data
    );
    switch errorData {
    | None => 0.0
    | Some(errorData) => {
      let (weightErrors, errorSum) = errorData;
      let weightErrors = linearClip(config.gradientClip, weightErrors);
      ignore(updateWeights(net, weightErrors, config.alpha));
      errorSum;
    }
  }
};

let trainEpoch = fun(config: trainingConfig, net: network, data: trainingData) {
  let finalDataArray = config.shuffle ? shuffleArray(Array.of_list(data)) : Array.of_list(data);
  let dataLength = Array.length(finalDataArray);
  let miniBatchSize = switch config.batchSize {
    | 0 => dataLength
    | _ => config.batchSize
  };
  let (error, _) = Array.fold_left(
    (iterDetails, _) => {
      let (error, lastIndex) = iterDetails;
      let thisBatchSize = min(miniBatchSize, dataLength - lastIndex);
      let newError = error +. trainMiniBatch(config, net, Array.to_list(Array.sub(finalDataArray, lastIndex, thisBatchSize)));
      (newError, lastIndex + thisBatchSize);
    },
    (0.0, 0),
    Array.make(1 + dataLength / miniBatchSize, 0)
  );
  error;
};

let setLayerTransfer = fun(net: network, layerInd: int, transfer: transferFunc) {
  ignore(Array.map(
    (perc) => perc.transfer = transfer,
    net[layerInd]
  ));
};

let trainNetwork = fun(config: trainingConfig, net: network, data: trainingData) {
  let error = ref(config.targetError +. 1.0);
  let epochs = ref(0);

  while(epochs^ < config.maxEpochs && error^ > config.targetError) {
    epochs := epochs^ + 1;
    error := trainEpoch(config, net, data);
    Js.log("Epoch " ++ string_of_int(epochs^) ++ ": error is " ++ string_of_float(error^));
  };

  net;
};

let myNet = makeNetwork(relu, [5, 1]);
setLayerTransfer(myNet, 1, linear);
let data = makeSingleValFunctionData((inp) => Pervasives.abs_float(Pervasives.sin(inp)), -5.0, 5.0, 100);
let config = {
  shuffle: true,
  gradientClip: 10000.0,
  maxEpochs: 1000,
  targetError: 0.1,
  alpha: 0.001,
  batchSize: 20
};

trainNetwork(config, myNet, data);
