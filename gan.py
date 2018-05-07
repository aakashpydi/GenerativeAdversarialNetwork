from datasets import DatasetSetup
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as torch_datasets

import torchvision.utils as vutils
from torch.autograd import Variable


### First need to define the generator and discriminator nerural nets
### architecture --> https://medium.com/@awjuliani/generative-adversarial-networks-explained-with-a-classic-spongebob-squarepants-episode-54deab2fce39
class GeneratorNN(nn.Module):

    def __init__(self):
        super(GeneratorNN, self).__init__()
        # layer_1_size = 1024
        input_layer_node_count = 100
        layer_1_node_count = 512
        layer_2_node_count = 256
        layer_3_node_count = 128
        layer_4_node_count = 64
        output_layer_node_count = 3

        self.generator_nn_model = nn.Sequential(
            nn.ConvTranspose2d(input_layer_node_count, layer_1_node_count, 4, 1, 0, bias = False),
            nn.BatchNorm2d(layer_1_node_count),
            nn.ReLU(True),
            nn.ConvTranspose2d(layer_1_node_count, layer_2_node_count, 4, 2, 1, bias = False),
            nn.BatchNorm2d(layer_2_node_count),
            nn.ReLU(True),
            nn.ConvTranspose2d(layer_2_node_count, layer_3_node_count, 4, 2, 1, bias = False),
            nn.BatchNorm2d(layer_3_node_count),
            nn.ReLU(True),
            nn.ConvTranspose2d(layer_3_node_count, layer_4_node_count, 4, 2, 1, bias = False),
            nn.BatchNorm2d(layer_4_node_count),
            nn.ReLU(True),
            nn.ConvTranspose2d(layer_4_node_count, output_layer_node_count, 4, 2, 1, bias = False),
            nn.Tanh()
        )

        self.loss_function = nn.BCELoss()
        self.optimizer = optim.Adam(self.generator_nn_model.parameters(), lr = 0.0002, betas = (0.5, 0.999))

    def forward(self, input):
        result = self.generator_nn_model(input)
        return result

class DiscriminatorNN(nn.Module):

    def __init__(self):
        input_layer_node_count = 3
        layer_1_node_count = 64
        layer_2_node_count = 128
        layer_3_node_count = 256
        layer_4_node_count = 512
        output_layer_node_count = 1
        leaky_relu_negative_slope = 0.2

        super(DiscriminatorNN, self).__init__()
        self.discriminator_nn_model = nn.Sequential(
            nn.Conv2d(input_layer_node_count, layer_1_node_count, 4, 2, 1, bias = False),
            nn.LeakyReLU(leaky_relu_negative_slope, inplace = True),
            nn.Conv2d(layer_1_node_count, layer_2_node_count, 4, 2, 1, bias = False),
            nn.BatchNorm2d(layer_2_node_count),
            nn.LeakyReLU(leaky_relu_negative_slope, inplace = True),
            nn.Conv2d(layer_2_node_count, layer_3_node_count, 4, 2, 1, bias = False),
            nn.BatchNorm2d(layer_3_node_count),
            nn.LeakyReLU(leaky_relu_negative_slope, inplace = True),
            nn.Conv2d(layer_3_node_count, layer_4_node_count, 4, 2, 1, bias = False),
            nn.BatchNorm2d(layer_4_node_count),
            nn.LeakyReLU(leaky_relu_negative_slope, inplace = True),
            nn.Conv2d(layer_4_node_count, output_layer_node_count, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

        self.loss_function = nn.BCELoss()
        self.optimizer = optim.Adam(self.discriminator_nn_model.parameters(), lr = 0.0002, betas = (0.5, 0.999))

    def forward(self, input):
        result = self.discriminator_nn_model(input)
        #print(result)
        return result.view(-1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

####################################### MAIN SCRIPT BEGINS ################################################################

############# Existing Model Image Generation Script ###################
STATE_SAVE_RAVI_FILE = 'ravi_saved_state.tar'
STATE_SAVE_BAPU_FILE = 'bapu_saved_state.tar'
STATE_SAVE_COMBINED_FILE = 'combined_saved_state.tar'
STATE_SAVE_CIFAR10_FILE = 'cifar10_saved_state.tar'

isRaviExists = os.path.isfile(STATE_SAVE_RAVI_FILE)
isBapuExists = os.path.isfile(STATE_SAVE_BAPU_FILE)
isCombinedExists = os.path.isfile(STATE_SAVE_COMBINED_FILE)
isCifar10Exists = os.path.isfile(STATE_SAVE_CIFAR10_FILE)


if(isRaviExists):
    state_ravi = torch.load(STATE_SAVE_RAVI_FILE)
    ravi_generator_nn = GeneratorNN()
    ravi_generator_nn.load_state_dict(state_ravi['generator_nn_state_dict'])
    print "Trained GAN Model for Ravi Varma paintings exists."
if(isBapuExists):
    state_bapu = torch.load(STATE_SAVE_BAPU_FILE)
    bapu_generator_nn = GeneratorNN()
    bapu_generator_nn.load_state_dict(state_bapu['generator_nn_state_dict'])
    print "Trained GAN Model for Bapu paintings exists."
if(isCombinedExists):
    state_combined = torch.load(STATE_SAVE_COMBINED_FILE)
    combined_generator_nn = GeneratorNN()
    combined_generator_nn.load_state_dict(state_combined['generator_nn_state_dict'])
    print "Trained GAN Model for Combined (Ravi+Bapu) exists."
if(isCifar10Exists):
    state_cifar10 = torch.load(STATE_SAVE_CIFAR10_FILE)
    cifar10_generator_nn = GeneratorNN()
    cifar10_generator_nn.load_state_dict(state_cifar10['generator_nn_state_dict'])
    print "Trained GAN Model for CIFAR10 exists."



if(isRaviExists or isBapuExists or isCombinedExists or isCifar10Exists):
    input_string = "Enter Corresponding Number.\n1 -> Generate and Save Image from Bapu Model\n2 -> Generate and Save Image from Ravi Model\n"
    input_string += "3 -> Generate and Save Image from Combined Model.\n4 -> Generate and Save from Cifar10 Model\n5--> Exit and Train New Model\n"
    while(True):
        random_noise_wrapped = Variable(torch.randn(32, 100, 1, 1))
        enteredValue = input(input_string)
        if enteredValue == 1 and isBapuExists:
            generated_image_set = bapu_generator_nn(random_noise_wrapped)
            vutils.save_image(generated_image_set.data, 'bapu_generated_image.png', normalize = True)
        elif enteredValue == 2 and isRaviExists:
            generated_image_set = ravi_generator_nn(random_noise_wrapped)
            vutils.save_image(generated_image_set.data, 'ravi_generated_image.png', normalize = True)
        elif enteredValue == 3 and isCombinedExists:
            generated_image_set = combined_generator_nn(random_noise_wrapped)
            vutils.save_image(generated_image_set.data, 'combined_generated_image.png', normalize = True)
        elif enteredValue == 4 and isCifar10Exists:
             generated_image_set = cifar10_generator_nn(random_noise_wrapped)
             vutils.save_image(generated_image_set.data, 'cifar10_generated_image.png', normalize = True)
        elif enteredValue == 5:
            print "Exiting Trained Model Image Generation Script.\nTraining Script Commencing.\n\n"
            break
        else:
            print "Illegal Input or Model Doesn't Exist (check current directory)"
else:
    print "No trained GAN model exists."


############# New Model Training Script ###################
EPOCH_COUNT = 500
BATCH_SIZE = 32

STATE_SAVE_FILE_NAME = 'saved_state.tar'

if os.path.isfile(STATE_SAVE_FILE_NAME):

    train_data_set = DatasetSetup(image_size=64, batch_size=BATCH_SIZE)
    #train_data_set.initRaviVarma()
    #train_data_set.initBapu()
    train_data_set.initCombined()
    ##train_data_set.initCIFAR10()

    state = torch.load(STATE_SAVE_FILE_NAME)
    generator_nn = GeneratorNN()
    generator_nn.load_state_dict(state['generator_nn_state_dict'])

    discriminator_nn = DiscriminatorNN()
    discriminator_nn.load_state_dict(state['discriminator_nn_state_dict'])

    start_epoch = state['completed_epoch'] + 1
    inner_loop_start = 0

    discriminator_loss_list = state['d_loss_list']
    generator_loss_list = state['g_loss_list']
    iteration_time_list = state['i_time_list']

    print "Found Saved State File. Commencing training from epoch: " + str(start_epoch)

else:
    start_epoch = 0
    inner_loop_start = 0

    train_data_set = DatasetSetup(image_size=64, batch_size=BATCH_SIZE)
    #train_data_set.initRaviVarma()
    #train_data_set.initBapu()
    train_data_set.initCombined()
    #train_data_set.initCIFAR10()

    generator_nn = GeneratorNN()
    generator_nn.apply(weights_init)

    discriminator_nn = DiscriminatorNN()
    discriminator_nn.apply(weights_init)

    discriminator_loss_list = list()
    generator_loss_list = list()
    iteration_time_list = list()

    print "No Saved State File Found. Commencing training from scratch.\n\n"

for epoch in range(start_epoch, EPOCH_COUNT):

    for i, data_tuple in enumerate(train_data_set.train_data_loader, inner_loop_start):
        status =  "EPOCH: [" +str(epoch) + "/" + str(EPOCH_COUNT) + "], \tIteration: [" + str(i) +"/" +str(train_data_set.total_batch_count) + "]"
        print(status)

        iteration_start_time = time.time()

        discriminator_nn.zero_grad()

        real_image_set, _ = data_tuple

        input_data_wrapped = Variable(real_image_set)
        #### we set the expected target to 1 (TRUE). The discriminator thereby learns to classify real images as 1 (TRUE)
        target_wrapped = Variable(torch.ones(input_data_wrapped.size()[0]))
        feed_forward_output = discriminator_nn(input_data_wrapped)
        #print(target_wrapped)
        #print(feed_forward_output)

        discriminator_loss_train_image_set = discriminator_nn.loss_function(feed_forward_output , target_wrapped)
        #print str(input_data_wrapped.size()[0])


        random_noise_wrapped = Variable(torch.randn(input_data_wrapped.size()[0], 100, 1, 1))
        generated_image_set = generator_nn(random_noise_wrapped)
        #### we set the expected target to 0 (FALSE). The discriminator thereby learns to classify the image
        #### produced by the generator as 0 (FALSE)
        target_wrapped = Variable(torch.zeros(input_data_wrapped.size()[0]))
        feed_forward_output = discriminator_nn(generated_image_set.detach())
        discriminator_loss_generated_image_set = discriminator_nn.loss_function(feed_forward_output, target_wrapped)
        #print(discriminator_loss_generated_image_set)

        #### we allow discriminator to learn on loss from generated image set if loss is above threshold value
        if discriminator_loss_generated_image_set.data[0] > 0.45:
            discriminator_loss_generated_image_set.backward()
            discriminator_nn.optimizer.step()

        generator_nn.zero_grad()
        target_wrapped = Variable(torch.ones(input_data_wrapped.size()[0]))
        #### we set the expected target to 1 (TRUE). The generator thereby learns to classify the output of
        #### the discriminator as 1 (TRUE)
        feed_forward_output = discriminator_nn(generated_image_set)
        generator_loss = generator_nn.loss_function(feed_forward_output , target_wrapped)

        if generator_loss.data[0] > 0.3:
            generator_loss.backward()
            generator_nn.optimizer.step()

        if discriminator_loss_train_image_set.data[0] > 0.35:
            discriminator_loss_train_image_set.backward()
            discriminator_nn.optimizer.step()

        iteration_time = time.time() - iteration_start_time

        iteration_time_list.append(iteration_time)
        discriminator_loss_list.append(discriminator_loss_generated_image_set.data[0])
        generator_loss_list.append(generator_loss.data[0])


        if i % 10 == 0:
            if os.path.isdir(folder_name) != True:
                #print("EPOCH FOLDER DOESN'T EXIST")
                os.makedirs(folder_name)
            file_name = "./output/EPOCH_" + str(epoch) + "/generated_image_" +str(i)+".png"
            vutils.save_image(generated_image_set.data, file_name , normalize = True)


        if i == 0:
            file_name = "./output/EPOCH_" + str(epoch) + "/real_image_" +str(i)+".png"
            vutils.save_image(real_image_set, file_name, normalize = True)

    state_to_save = {
                'generator_nn_state_dict': generator_nn.state_dict(),
                'discriminator_nn_state_dict': discriminator_nn.state_dict(),
                'completed_epoch': epoch,
                'd_loss_list': discriminator_loss_list,
                'g_loss_list' : generator_loss_list,
                'i_time_list' : iteration_time_list
                }
    torch.save(state_to_save, STATE_SAVE_FILE_NAME)
