type filenames = {
  images: string,
  labels: string
};

type imageDimensions = {
  x: int,
  y: int
};

type example = {
  data: Buffer.t,
  label: int
};

exception CorruptedExampleFile(string);

let readExamples = fun(files: filenames) {
  let imageFile = open_in(files.images);
  let labelFile = open_in(files.labels);

  if (input_binary_int(imageFile) !== 2051) {
    raise(CorruptedExampleFile("Magic number is incorrect"));
  };
  if (input_binary_int(labelFile) !== 2049) {
    raise(CorruptedExampleFile("Magic number is incorrect"));
  };

  let labelCount = input_binary_int(labelFile);
  let imageCount = input_binary_int(imageFile);
  if (labelCount !== imageCount) {
    raise(CorruptedExampleFile("Label file and image file have different numbers of examples"));
  };

  let imageDims = {
    x: input_binary_int(imageFile),
    y: input_binary_int(imageFile)
  }
  let imageSize = imageDims.x * imageDims.y;

  let examples = ref([]);

  for (i in 0 to imageCount) {
    let buf = Buffer.create(imageSize);
    input(imageFile, buf, 0, imageSize);
    let label = input_byte(labelFile);

    examples := [{ data: buf, label: label }, ...examples^];
  };

  examples;
};
