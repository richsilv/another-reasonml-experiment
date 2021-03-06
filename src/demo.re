Random.init(1);
let updateSeed = fun(newSeed: int) => Random.init(newSeed);
Arg.parse([("-seed", Int(updateSeed), "Use the specified seed")], (_: string) => (), "Neural network");

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

type trainingConfig = {
  shuffle: bool,
  gradientClip: float,
  maxEpochs: int,
  targetError: float,
  gamma: float,
  batchSize: int,
  decreasedErrorAlpha: float,
  increasedErrorAlpha: float,
  minAlpha: float,
  mutable alpha: float,
};

let makeWeights = fun(length: int) => {
  Array.map(
    fun(_) => Random.float(1.0),
    Array.make(length, 0.0));
};

let print_string_ln = fun(str: string) => {
  print_string(str ++ "\n");
};

let softplus = {
  func: fun(inp: float) => {
    /* Softplus is not numerically stable for large x, but it's
       also arbitrarily close to the identity fn */
    if (inp > 32.0) {
      inp;
    } else {
      log(1.0 +. exp(inp));
    };
  },
  deriv: fun(inp: float) => {
    1.0 /. (1.0 +. exp(-. inp));
  }
};

let reluNegGrad = 0.01;

let relu = {
  func: fun(inp: float) => {
    if (inp >= 0.0) {
      inp;
    } else {
      reluNegGrad *. inp;
    };
  },
  deriv: fun(inp: float) => {
    if (inp >= 0.0) {
      1.0
    } else {
      reluNegGrad;
    }
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
      let n = Array.fold_right(
        (x, y) => x +. y,
        Array.mapi(
          (ind, inputNode) => {
            switch inputNode {
              | Perceptron(thisPerc) => {
                let outp = calcOutput(thisPerc) *. perc.weights[ind];
                outp;
              };
              | Input(inp) => {
                let outp = inp *. perc.weights[ind];
                outp;
              }
            };
          },
          perc.inputNodes
        ),
        0.0
      );
      perc.output = perc.transfer.func(n);
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

type linearObj = Vector(array(float)) | Matrix(array(linearObj));

type network = {
  perceptrons: array(array(perceptron)),
  config: trainingConfig,
  mutable velocity: option(linearObj),
  mutable error: float
};

let rec recursivelyMakePerceptrons = fun(config: trainingConfig, transfer: transferFunc, shape: list(int), perceptrons: option(array(array(perceptron)))) => {
  let inputs = switch perceptrons {
    | None => {
      Array.make(List.hd(shape), Input(0.0));
    }
    | Some(perceptrons) => {
      Array.map(
        (perc) => Perceptron(perc),
        perceptrons[Array.length(perceptrons) - 1]
      );
    }
  };

  let existing = switch perceptrons {
    | None => [||]
    | Some(perceptrons) => perceptrons
  };

  switch shape {
    | [] => {
      existing;
    }
    | [layerSize, ...layers] => {
      recursivelyMakePerceptrons(
        config,
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

let makeNetwork = fun(config: trainingConfig, tranfer: transferFunc, shape: list(int)) => {
  {
    perceptrons: recursivelyMakePerceptrons(config, tranfer, shape, None),
    config: config,
    velocity: None,
    error: infinity
  };
};

let calcOutput = fun(net: network) => {
  let outputLayer = net.perceptrons[Array.length(net.perceptrons) - 1];
  Array.map(
    (perc) => calcOutput(perc),
    outputLayer
  );
};

let setInputs = fun(net: network, inputs: networkInputs) => {
  let inputLayer = net.perceptrons[0];
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
        print_string_ln("WARNING - you are connecting perceptrons rather than values as network inputs");
      };
    },
    inputLayer
  ));
  ignore(Array.iter(
    (layer) => Array.iter(
      (perc) => perc.invalid = true,
      layer
    ),
    net.perceptrons
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
  print_string_ln("Inputs");
  ignore(Array.iter(
    (inputNode) => switch inputNode {
      | Perceptron(_) => print_string_ln("Perceptron")
      | Input(inp) => print_float(inp)
    },
    net.perceptrons[0][0].inputNodes
  ));
  Array.mapi(
    (ind, layer) => {
      print_string_ln("Layer " ++ string_of_int(ind));
      Array.map(
        (perc) => print_float(perc.output),
        layer
      )
    },
    net.perceptrons
  );
};

let logWeights = fun(net: network) {
  ignore(Array.mapi(
    (ind, layer) => {
      print_string_ln("Layer " ++ string_of_int(ind));
      Array.map(
        (perc) => print_string_ln(Array.fold_left(
          (soFar, weight) => soFar ++ " " ++ string_of_float(weight),
          "",
          perc.weights
        )),
        layer
      )
    },
    net.perceptrons
  ));
};

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
      print_string_ln(leftPad(Array.fold_left(
        (output, x) => output ++ " " ++ string_of_float(x),
        leftPad("", indent),
        v
      ), indent));
    }
    | Matrix(m) => {
      Array.iteri(
        (ind, innerObj) => {
          print_string_ln(leftPad(string_of_int(ind), indent));
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

let rec linearScalarProd = fun(a: linearObj, p: float): linearObj {
  switch a {
    | Vector(u) => {
      Vector(Array.map(
        (entry) => entry *. p,
        u
      ));
    }
    | Matrix(m) => {
      Matrix(Array.map(
        (entry) => linearScalarProd(entry, p),
        m
      ));
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
  let reversed = Array.of_list(List.rev(Array.to_list(net.perceptrons)));
  Matrix(Array.of_list(List.rev(Array.to_list(Array.map(
    (layer) => {
      Matrix(Array.mapi(
        (ind, thisPerc) => {
          let dEdO = switch thisPerc.outputNodes {
            | Output(output) => {
              let thisExpectedOutput = expectedOutput[ind];
              switch thisExpectedOutput {
                | Output(floatOutput) => output -. floatOutput
                | Perceptrons(_) => raise(WrongNodeType("Perceptron"))
              };
            }
            | Perceptrons(outputPercs) => {
              sum(Array.map(
                (outputPerc) => {
                  /*  Here we are using the index of THIS neuron to access the weight index of this neuron's
                      input into the output neuron. */
                  outputPerc.dEdN *. outputPerc.weights[ind]
                },
                outputPercs
              ));
            }
          };
          let dOdN = thisPerc.transfer.deriv(thisPerc.output);
          thisPerc.dEdN = dEdO *. dOdN;
          Vector(Array.map(
            (inputNode) => {
              thisPerc.dEdN *. switch inputNode {
                | Perceptron(inputPerc) => inputPerc.output
                | Input(float) => float
              };
            },
            thisPerc.inputNodes
          ));
        },
        layer
      ));
    },
    reversed
  )))));
};

let updateWeights = fun(net: network, velocity: linearObj) {
  switch velocity {
    | Matrix(layers) => {
      ignore(Array.mapi(
        (indA, layer) => {
          switch layer {
            | Matrix(percEntries) => {
              ignore(Array.mapi(
                (indB, percEntry) => {
                  switch percEntry {
                    | Vector(weightUpdates) => {
                      ignore(Array.mapi(
                        (indC, weightUpdate) => {
                          let perc = net.perceptrons[indA][indB];
                          let currentWeight = perc.weights[indC];
                          Array.set(perc.weights, indC, currentWeight -. weightUpdate);
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
                        weightUpdates
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
  };
  net.velocity = Some(velocity);
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

let shuffleArray = fun(a: array('x)) {
  let b = Array.copy(a);
  Array.fast_sort(
    (_, _) => Random.int(2) - 1,
    b
  );
  b;
};

let trainMiniBatch = fun(net: network, data: trainingData) {
  let errorData = List.fold_left(
    (errorSoFar, example) => Some(trainExample(net, errorSoFar, example)),
    None,
    data
  );
  switch errorData {
    | None => 0.0;
    | Some(errorData) => {
      let (weightDerivs, errorSum) = errorData;
      let newVelocity = switch net.velocity {
        | None => {
          linearClip(net.config.gradientClip, linearScalarProd(weightDerivs, net.config.alpha));
        }
        | Some(velocity) => {
          linearClip(net.config.gradientClip, linearAdd(
            linearScalarProd(velocity, net.config.gamma),
            linearScalarProd(weightDerivs, (1.0 -. net.config.gamma) *. net.config.alpha /. float_of_int(net.config.batchSize))
          ));
        }
      };
      ignore(updateWeights(net, newVelocity));
      errorSum;
    }
  }
};

let trainEpoch = fun(net: network, data: trainingData) {
  let finalDataArray = net.config.shuffle ? shuffleArray(Array.of_list(data)) : Array.of_list(data);
  let dataLength = Array.length(finalDataArray);
  let miniBatchSize = switch net.config.batchSize {
    | 0 => dataLength
    | _ => net.config.batchSize
  };
  let (error, _) = Array.fold_left(
    (iterDetails, _) => {
      let (errorSoFar, lastIndex) = iterDetails;
      let thisBatchSize = min(miniBatchSize, dataLength - lastIndex);
      let thisError = trainMiniBatch(net, Array.to_list(Array.sub(finalDataArray, lastIndex, thisBatchSize)));
      let newError = errorSoFar +. thisError;
      (newError, lastIndex + thisBatchSize);
    },
    (0.0, 0),
    Array.make(1 + dataLength / miniBatchSize, 0)
  );
  if (error < net.error) {
    net.config.alpha = net.config.alpha *. net.config.decreasedErrorAlpha;
  } else {
    net.config.alpha = max(net.config.alpha *. net.config.increasedErrorAlpha, net.config.minAlpha);
  };
  net.error = error;
  error;
};

let setLayerTransfer = fun(net: network, layerInd: int, transfer: transferFunc) {
  ignore(Array.map(
    (perc) => perc.transfer = transfer,
    net.perceptrons[layerInd]
  ));
};

let trainNetwork = fun(net: network, data: trainingData) {
  let error = ref(net.config.targetError +. 1.0);
  let epochs = ref(0);

  while(epochs^ < net.config.maxEpochs && error^ > net.config.targetError) {
    let thisError = trainEpoch(net, data);
    epochs := epochs^ + 1;
    error := thisError;
    print_string_ln("Epoch " ++ string_of_int(epochs^) ++ ": error is " ++ string_of_float(error^) ++ "; Alpha is " ++ string_of_float(net.config.alpha));
  };

  net;
};

let data = makeSingleValFunctionData((inp) => Pervasives.abs_float(Pervasives.sin(inp)), -5.0, 5.0, 500);
let config = {
  shuffle: true,
  gradientClip: 10000.0,
  maxEpochs: 10000,
  targetError: 0.1,
  alpha: 0.001,
  minAlpha: 0.00000001,
  increasedErrorAlpha: 0.5,
  decreasedErrorAlpha: 1.05,
  gamma: 0.5,
  batchSize: 100
};
let myNet = makeNetwork(config, tanhAct, [5, 10, 15, 10, 5, 1]);
setLayerTransfer(myNet, 1, relu);
setLayerTransfer(myNet, 3, relu);
setLayerTransfer(myNet, 5, linear);

trainNetwork(myNet, data);
