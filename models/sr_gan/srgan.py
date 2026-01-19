import keras
import tensorflow as tf
from keras import Model
from keras.applications.vgg19 import preprocess_input as vgg_preprocess


class SRGAN(Model):
    def __init__(  # noqa: PLR0913
        self,
        generator,
        discriminator,
        vgg,
        content_weight=1.0,
        adversarial_weight=1e-3,
        pixel_weight=1.0,
        tv_weight = 0.,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.vgg = vgg
        self.content_weight = content_weight
        self.adversarial_weight = adversarial_weight
        self.pixel_weight = pixel_weight
        self.tv_weight = tv_weight

        # metrics
        self.g_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.psnr_metric = tf.keras.metrics.Mean(name="psnr_metric")

    def call(self, lr, *, training=False):
        return self.generator(lr, training=training)

    @property
    def metrics(self):
        return [self.g_loss_tracker, self.d_loss_tracker, self.psnr_metric]

    def compile(self, g_optimizer, d_optimizer, content_loss_fn, adv_loss_fn, pixel_loss_fn):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.content_loss_fn = content_loss_fn
        self.adv_loss_fn = adv_loss_fn
        self.pixel_loss_fn = pixel_loss_fn

    def compute_vgg_features(self, hr):
        hr_vgg = vgg_preprocess(hr * 255.0)
        return self.vgg(hr_vgg)

    def train_step(self, data):
        lr, hr = data

        real_labels = tf.ones((tf.shape(hr)[0], 1)) * 0.9   # disc label smoothening, should help avoiding overconfidence
        fake_labels = tf.zeros((tf.shape(hr)[0], 1))

        # generate fake images
        fake_hr = self.generator(lr, training=True)

        with tf.GradientTape() as tape_d:
            d_real = self.discriminator(hr, training=True)
            d_fake = self.discriminator(fake_hr, training=True)

            d_loss_real = self.adv_loss_fn(real_labels, d_real)
            d_loss_fake = self.adv_loss_fn(fake_labels, d_fake)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        grads_d = tape_d.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads_d, self.discriminator.trainable_variables, strict=True))

        with tf.GradientTape() as tape_g:
            fake_hr = self.generator(lr, training=True)
            d_fake_for_g = self.discriminator(fake_hr, training=False)

            # perceptual loss with VGG
            vgg_fake = self.compute_vgg_features(fake_hr)
            vgg_real = self.compute_vgg_features(hr)
            vgg_loss = self.content_loss_fn(vgg_real, vgg_fake)

            adv_loss = self.adv_loss_fn(real_labels, d_fake_for_g)

            pixel_loss = self.pixel_loss_fn(hr, fake_hr)
            
            tv_loss = (
                tf.reduce_mean(tf.image.total_variation(fake_hr))
                if self.tv_weight and self.tv_weight > 0.0
                else 0.0
                )   # fmt: skip
            
            g_loss = (
                self.content_weight * vgg_loss
                + self.adversarial_weight * adv_loss
                + self.pixel_weight * pixel_loss
                + self.tv_weight * tv_loss
            )   # fmt: skip

        grads_g = tape_g.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads_g, self.generator.trainable_variables, strict=True))

        # metrics
        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)
        # psnr between real and fake
        psnr_val = tf.reduce_mean(tf.image.psnr(tf.clip_by_value(fake_hr, 0.0, 1.0), hr, max_val=1.0))
        self.psnr_metric.update_state(psnr_val)

        return {
            "loss": self.g_loss_tracker.result(),
            "d_loss": self.d_loss_tracker.result(),
            "psnr_metric": self.psnr_metric.result(),
        }

    def test_step(self, data):
        lr, hr = data

        fake_hr = self.generator(lr, training=False)

        # discriminator outputs
        d_real = self.discriminator(hr, training=False)
        d_fake = self.discriminator(fake_hr, training=False)

        # labels
        real_labels = tf.ones((tf.shape(hr)[0], 1)) * 0.9
        fake_labels = tf.zeros((tf.shape(hr)[0], 1))

        # losses
        d_loss_real = self.adv_loss_fn(real_labels, d_real)
        d_loss_fake = self.adv_loss_fn(fake_labels, d_fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # content/pixel/adversarial for generator evaluation
        vgg_fake = self.compute_vgg_features(fake_hr)
        vgg_real = self.compute_vgg_features(hr)
        vgg_loss = self.content_loss_fn(vgg_real, vgg_fake)
        adv_loss_for_g = self.adv_loss_fn(real_labels, d_fake)
        pixel_loss = self.pixel_loss_fn(hr, fake_hr)

        tv_loss = (
                tf.reduce_mean(tf.image.total_variation(fake_hr))
                if self.tv_weight and self.tv_weight > 0.0
                else 0.0
                )   # fmt: skip
        
        g_loss = (
            self.content_weight * vgg_loss
            + self.adversarial_weight * adv_loss_for_g
            + self.pixel_weight * pixel_loss
            + self.tv_weight * tv_loss
        )

        # update metrics
        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)
        psnr_val = tf.reduce_mean(tf.image.psnr(tf.clip_by_value(fake_hr, 0.0, 1.0), hr, max_val=1.0))
        self.psnr_metric.update_state(psnr_val)

        return {
            "loss": self.g_loss_tracker.result(),
            "d_loss": self.d_loss_tracker.result(),
            "psnr_metric": self.psnr_metric.result(),
        }
