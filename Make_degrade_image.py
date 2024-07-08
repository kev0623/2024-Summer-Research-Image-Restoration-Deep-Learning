import cv2
import numpy as np

# Define the image name variable
image_name = 'CameraMen'
image_type = '.png'
Input_image_name = 'Photos/'+image_name+ image_type 
Output_image_name = 'Degrade_Photos/'+image_name+'_'


def add_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype('uint8')
    noisy_image = cv2.add(image, gauss)
    return noisy_image


image = cv2.imread(Input_image_name)
noisy_image = add_gaussian_noise(image)
cv2.imwrite(Output_image_name+'noisy_image_gaussian'+ image_type , noisy_image)


def add_poisson_noise(image):
    image_float = image.astype(float)
    noisy_image = np.random.poisson(image_float).astype(np.uint8)
    return noisy_image


image = cv2.imread(Input_image_name)
noisy_image = add_poisson_noise(image)
cv2.imwrite(Output_image_name+'noisy_image_poisson'+ image_type , noisy_image)


def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    noisy_image = image.copy()
    total_pixels = image.size // image.shape[2]

    # Add salt noise
    num_salt = int(total_pixels * salt_prob)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 255

    # Add pepper noise
    num_pepper = int(total_pixels * pepper_prob)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 0

    return noisy_image


image = cv2.imread(Input_image_name)
noisy_image = add_salt_and_pepper_noise(image)
cv2.imwrite(Output_image_name+'noisy_image_salt_and_pepper'+image_type , noisy_image)


def add_speckle_noise(image):
    noise = np.random.randn(*image.shape) * 0.2
    noisy_image = image + image * noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


image = cv2.imread(Input_image_name)
noisy_image = add_speckle_noise(image)
cv2.imwrite(Output_image_name+'noisy_image_speckle' + image_type , noisy_image)


def add_local_variance_noise(image, variance_factor=0.1):
    row, col, ch = image.shape
    local_var_noise = np.zeros((row, col, ch), dtype=np.float32)
    for i in range(row):
        for j in range(col):
            local_variance = variance_factor * image[i, j]
            local_var_noise[i, j] = image[i, j] + \
                np.random.normal(0, local_variance)

    noisy_image = np.clip(local_var_noise, 0, 255).astype(np.uint8)
    return noisy_image


image = cv2.imread(Input_image_name)
noisy_image = add_local_variance_noise(image)
cv2.imwrite(Output_image_name+'noisy_image_local_variance' + image_type , noisy_image)


def add_uniform_noise(image, low=-10, high=10):
    noise = np.random.uniform(low, high, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


image = cv2.imread(Input_image_name)
noisy_image = add_uniform_noise(image)
cv2.imwrite(Output_image_name+'noisy_image_uniform' + image_type , noisy_image)


def add_impulse_noise(image, amount=0.05):
    noisy_image = image.copy()
    num_impulses = int(amount * image.size)
    coords = [np.random.randint(0, i - 1, num_impulses) for i in image.shape]
    noisy_image[coords[0], coords[1], coords[2] %
                3] = np.random.randint(0, 256, num_impulses)
    return noisy_image


image = cv2.imread(Input_image_name)
noisy_image = add_impulse_noise(image)
cv2.imwrite(Output_image_name+'noisy_image_impulse' + image_type , noisy_image)


def add_periodic_noise(image, frequency=5):
    row, col, ch = image.shape
    noise = np.zeros((row, col, ch), dtype=np.uint8)

    for i in range(row):
        for j in range(col):
            noise[i, j] = 128 + 127 * np.sin(2 * np.pi * frequency * i / row)

    noisy_image = cv2.add(image, noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


image = cv2.imread(Input_image_name)
noisy_image = add_periodic_noise(image)
cv2.imwrite(Output_image_name+'noisy_image_periodic' + image_type , noisy_image)
