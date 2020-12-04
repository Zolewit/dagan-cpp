#include <torch/torch.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <direct.h> //mkdir
#include <io.h>     //access

// The size of the noise vector fed to the generator.
const int64_t kNoiseSize = 1000;

// The batch size for training.
const int64_t kBatchSize = 128;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 30;

// Where to find the MNIST dataset.
const char *kDataFolder = "C:\\Users\\User\\data\\mnist\\raw";
const char *kOutPath = "sample3\\";
// After how many batches to create a new checkpoint periodically.
const int64_t kCheckpointEvery = 450;

// How many images to sample at every checkpoint.
const int64_t kNumberOfSamplesPerCheckpoint = 10;

// Set to `true` to restore models and optimizers from previously saved
// checkpoints.
const bool kRestoreFromCheckpoint = false;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

using namespace torch;

struct DCGANGeneratorImpl : nn::Module
{
  DCGANGeneratorImpl(int kNoiseSize)
      : conv1(nn::ConvTranspose2dOptions(kNoiseSize, 256, 4)
                  .bias(false)),
        batch_norm1(256),
        conv2(nn::ConvTranspose2dOptions(256, 128, 3)
                  .stride(2)
                  .padding(1)
                  .bias(false)),
        batch_norm2(128),
        conv3(nn::ConvTranspose2dOptions(128, 64, 4)
                  .stride(2)
                  .padding(1)
                  .bias(false)),
        batch_norm3(64),
        conv4(nn::ConvTranspose2dOptions(64, 1, 4)
                  .stride(2)
                  .padding(1)
                  .bias(false))
  {
    // register_module() is needed if we want to use the parameters() method later on
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    register_module("batch_norm1", batch_norm1);
    register_module("batch_norm2", batch_norm2);
    register_module("batch_norm3", batch_norm3);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::relu(batch_norm1(conv1(x)));
    x = torch::relu(batch_norm2(conv2(x)));
    x = torch::relu(batch_norm3(conv3(x)));
    x = torch::tanh(conv4(x));
    return x;
  }

  nn::ConvTranspose2d conv1, conv2, conv3, conv4;
  nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};

TORCH_MODULE(DCGANGenerator); //定义DCGANGeneratorImpl，封装为DCGANGenerator
//DCGANGenerator是Impl的智能指针

struct DiscriminatorImpl : nn::Module
{
  DiscriminatorImpl()
      : conv1(nn::Conv2dOptions(1, 64, 4)
                  .stride(2)
                  .padding(1)
                  .bias(false)),
        conv2(nn::Conv2dOptions(64, 128, 4)
                  .stride(2)
                  .padding(1)
                  .bias(false)),
        batch_norm1(128),
        conv3(nn::Conv2dOptions(128, 256, 4)
                  .stride(2)
                  .padding(1)
                  .bias(false)),
        batch_norm2(256),
        conv4(nn::Conv2dOptions(256, 1, 3)
                  .stride(1)
                  .padding(0)
                  .bias(false))
  {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    register_module("batch_norm1", batch_norm1);
    register_module("batch_norm2", batch_norm2);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::leaky_relu(conv1(x), 0.2);
    x = torch::leaky_relu(batch_norm1(conv2(x)), 0.2);
    x = torch::leaky_relu(batch_norm2(conv3(x)), 0.2);
    x = torch::sigmoid(conv4(x));
    return x;
  }

  nn::Conv2d conv1, conv2, conv3, conv4;
  nn::BatchNorm2d batch_norm1, batch_norm2;
};
TORCH_MODULE(Discriminator);

int main(int argc, const char *argv[])
{
  torch::manual_seed(1);

  if (access(kOutPath, 0) == -1)
    // if this folder not exist, create a new one.
    mkdir(kOutPath); // 返回 0 表示创建成功，-1 表示失败

  // Create the device we pass around based on whether CUDA is available.
  // torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available())
  {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  }
  else
    std::cout << "CUDA is not available! Training on CPU." << std::endl;

  DCGANGenerator generator(kNoiseSize);
  generator->to(device);

  // nn::Sequential discriminator(
  //     // Layer 1
  //     nn::Conv2d(
  //         nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
  //     nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
  //     // Layer 2
  //     nn::Conv2d(
  //         nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
  //     nn::BatchNorm2d(128),
  //     nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
  //     // Layer 3
  //     nn::Conv2d(
  //         nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
  //     nn::BatchNorm2d(256),
  //     nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
  //     // Layer 4
  //     nn::Conv2d(
  //         nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
  //     nn::Sigmoid());
  Discriminator discriminator;
  discriminator->to(device);

  // Assume the MNIST dataset is available under `kDataFolder`;
  auto dataset = torch::data::datasets::MNIST(kDataFolder)
                     .map(torch::data::transforms::Normalize<>(0.5, 0.5)) //归一化
                     .map(torch::data::transforms::Stack<>());            //沿着第一个维度堆叠成一个tensor
  const int64_t batches_per_epoch =
      std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));

  auto data_loader = torch::data::make_data_loader(
      std::move(dataset),
      torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

  torch::optim::Adam generator_optimizer(
      generator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));
  torch::optim::Adam discriminator_optimizer(
      discriminator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));

  if (kRestoreFromCheckpoint) //从已保存的pt恢复，但是进度会重新计算
  {
    torch::load(generator, "generator-checkpoint.pt");
    torch::load(generator_optimizer, "generator-optimizer-checkpoint.pt");
    torch::load(discriminator, "discriminator-checkpoint.pt");
    torch::load(
        discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
  }

  int64_t checkpoint_counter = 1;
  for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
  {
    int64_t batch_index = 0;
    for (torch::data::Example<> &batch : *data_loader)
    {
      // Train discriminator with real images.
      discriminator->zero_grad();                        //清空梯度信息
      torch::Tensor real_images = batch.data.to(device); //将数据搬到cuda上
      torch::Tensor real_labels =
          torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
      //真标签定为0.8到1.0，以使鉴别器训练更加健壮。这个技巧叫做label smoothing
      torch::Tensor real_output = discriminator->forward(real_images); //输入真实数据，输出结果
      torch::Tensor d_loss_real =
          torch::binary_cross_entropy(real_output, real_labels); //计算loss
      d_loss_real.backward();                                    //反向传播

      // Train discriminator with fake images.
      torch::Tensor noise =
          torch::randn({batch.data.size(0), kNoiseSize, 1, 1}, device);         //随机噪声
      torch::Tensor fake_images = generator->forward(noise);                    //输入噪声，输出结果
      torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);     //假标签定为0
      torch::Tensor fake_output = discriminator->forward(fake_images.detach()); //detach截断反向传播的梯度流
      torch::Tensor d_loss_fake =
          torch::binary_cross_entropy(fake_output, fake_labels);
      d_loss_fake.backward();

      torch::Tensor d_loss = d_loss_real + d_loss_fake;
      discriminator_optimizer.step(); //更新参数

      // Train generator.
      generator->zero_grad();
      fake_labels.fill_(1);
      fake_output = discriminator->forward(fake_images); //d对假数据的结果作为g的成绩
      torch::Tensor g_loss =
          torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();
      generator_optimizer.step();
      ++batch_index;
      if (batch_index % kLogInterval == 0)
      {
        std::printf(
            "\r[%I64d/%I64d][%I64d/%I64d] D_loss: %.4f | G_loss: %.4f",
            epoch,
            kNumberOfEpochs,
            batch_index,
            batches_per_epoch,
            d_loss.item<float>(),
            g_loss.item<float>());
      }

      if (batch_index % kCheckpointEvery == 0)
      {
        // Checkpoint the model and optimizer state.
        torch::save(generator, "generator-checkpoint.pt");
        torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
        torch::save(discriminator, "discriminator-checkpoint.pt");
        torch::save(
            discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
        // Sample the generator and save the images.
        torch::Tensor samples = generator->forward(torch::randn(
            {kNumberOfSamplesPerCheckpoint, kNoiseSize, 1, 1}, device));
        torch::save(
            (samples + 1.0) / 2.0,
            torch::str(kOutPath, "dcgan-sample-", checkpoint_counter, ".pt"));
        std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
      }
    }
  }

  std::cout << "\nTraining complete!" << std::endl;

  system("pause");
}