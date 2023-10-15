from abc import ABC, abstractmethod
import argparse
from typing import Generator
import cv2
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm, trange

W = 'Debug Window'
SMP = 'TA_TIs_' # shared memory prefix: Thread Art Thread Images

Point = tuple[float, float]

class ThreadArtInfo(ABC):
    def __init__(self, n_nails: int, debug: bool = False) -> None:
        self._nails = []
        self._n_nails = n_nails
        self.image_shape = None
        self.image = None
        self._debug = debug
        self.__mems = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for mem in self.__mems.values():
            mem.close()
            mem.unlink()
        return False

    def prepare(self, image_shape: tuple[int], thread_width: float = 0.01, thread_opacity: float = 0.5, prepare_thread_images: bool = True):
        self.image_shape = image_shape
        self._mask = self.get_mask(self.image_shape)
        self._nails = self.get_nails(self._n_nails)
        self._thread_width = thread_width * self.image_shape[0]
        self._thread_opacity = thread_opacity

        if prepare_thread_images:
            # Create all thread images
            for n0 in range(len(self._nails)):
                name = f'{SMP}{n0}'
                dummy = np.ones((len(self._nails) - n0, *self.image_shape), dtype=np.uint8)
                self.__mems[name] = shared_memory.SharedMemory(create=True, size=dummy.nbytes, name=name)

            # Current image of all used threads
            name = f'{SMP}img'
            dummy = np.ones(self.image_shape, dtype=np.float32)
            self.__mems[name] = shared_memory.SharedMemory(create=True, size=dummy.nbytes, name=name)

            with mp.Pool(mp.cpu_count()) as pool:
                _ = pool.starmap(self._create_thread_image,
                                [(self._nails[n0], self._nails[n1], n0, n1, True) for n0 in range(len(self._nails)) for n1 in range(n0, len(self._nails))],
                                chunksize=200)

            if self._debug:
                n0 = 0
                n1 = len(self._nails) // 2
                while True:
                    cv2.imshow(W, self._get_thread_image(n0, n1))
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        break
                    elif key == ord('a'):
                        n0 -= 1
                    elif key == ord('d'):
                        n0 += 1
                    elif key == ord('w'):
                        n1 += 1
                    elif key == ord('s'):
                        n1 -= 1
                    n0 %= len(self._nails)
                    n1 %= len(self._nails)

    def load(self, nails: str|list[int]) -> np.ndarray:
        if isinstance(nails, str):
            # load nails from file
            nails = list(np.loadtxt(nails, dtype=np.int32))
        assert max(nails) < len(self._nails), 'Nail index out of range'
        self.nail_order = nails

    def apply_loaded_sequence(self, show_progress: bool) -> np.ndarray:
        for i in self.generate_frames(show_progress):
            pass
        return self.output_image

    def generate_frames(self, show_progress: bool = False) -> Generator[np.ndarray, None, None]:
        img = np.ones(self.image_shape, np.float32)
        if show_progress:
            P = 'Progress'
            cv2.namedWindow(P, cv2.WINDOW_GUI_NORMAL)
            cv2.imshow(P, img)
            cv2.waitKey(1)

        last_nail = self.nail_order[0]
        for nail in tqdm(self.nail_order[1:]):
            img = self._combine_images(img, self._get_thread_image(last_nail, nail))
            last_nail = nail
            if show_progress:
                cv2.imshow(P, img)
                cv2.waitKey(1) # really quick animation
            self.output_image = img
            yield img

    def calculate(self, image: np.ndarray, start_nail: int = -1, show_progress: bool = False, early_end: bool = True) -> tuple[list[int], np.ndarray]:
        self.image = image.copy()
        if self.image.dtype == np.uint8:
            self.image = self.image.astype(np.float32) / 255.0
        assert self.image.min() >= 0 and self.image.max() <= 255, 'Image must be in range [0, 255]'
        assert self.image.shape == self.image_shape, 'Image shape does not match'
        assert self.image.ndim == 2, 'Image must be grayscale'
        if self._debug:
            image_debug = self.image.copy()
            for nail in self._nails:
                cv2.circle(image_debug, (int(nail[0]), int(nail[1])), 0, 255, -1)
            cv2.imshow(W, image_debug)
            cv2.waitKey(0)
        elif show_progress:
            P = 'Progress'
            cv2.namedWindow(P, cv2.WINDOW_GUI_NORMAL)
            cv2.imshow(P, self.image)
            cv2.waitKey(1)

        nail_order = []
        if start_nail == -1:
            start_nail = np.random.randint(0, len(self._nails))
        else:
            start_nail %= len(self._nails)
        nail_order.append(start_nail)

        img = np.ones(self.image_shape, np.float32)
        last_nail = start_nail
        last_loss = self._loss(img)
        with mp.Pool(mp.cpu_count()) as pool:
            with tqdm(desc='Calculating', unit=' conns', leave=True) as pbar:
                # greedy algorithm as described in "The Mathematics of String Art" by Virtually Passed (https://youtu.be/WGccIFf6MF8)
                while True:
                    best_nail, best_loss = self.__find_best_nail(img, last_nail, last_loss, pool)
                    pbar.update(1)
                    if best_nail is None:
                        if early_end:
                            break
                        # find best nail from current nail failed
                        # find best nail pair
                        best_last_nail = None
                        best_next_nail = None
                        best_loss = last_loss
                        for n in tqdm(range(len(self._nails)), total=len(self._nails), desc='Find best nail pair'):
                            if n == last_nail:
                                continue
                            best_nail2, best_loss2 = self.__find_best_nail(img, n, last_loss, pool)
                            if best_nail2 is not None and best_loss2 < best_loss:
                                best_last_nail = n
                                best_next_nail = best_nail2
                                best_loss = best_loss2
                        if best_last_nail is None:
                            # no nail pair found
                            print('Finished')
                            break
                        print('Found new nail pair', best_last_nail, best_next_nail)
                        # loop around to the best nail
                        if np.abs(best_last_nail - last_nail) < len(self._nails) / 2:
                            for n in range(last_nail, best_last_nail, 1 if best_last_nail > last_nail else -1):
                                nail_order.append(n)
                        else:
                            if best_last_nail > last_nail:
                                for n in range(last_nail, 0, -1):
                                    nail_order.append(n)
                                for n in range(len(self._nails)-1, best_last_nail, -1):
                                    nail_order.append(n)
                            else:
                                for n in range(last_nail, len(self._nails)):
                                    nail_order.append(n)
                                for n in range(0, best_last_nail):
                                    nail_order.append(n)
                        img = self._combine_images(img, self._get_thread_image(last_nail, best_last_nail))
                        nail_order.append(best_last_nail)
                        last_nail = best_last_nail
                        best_nail = best_next_nail

                    nail_order.append(best_nail)
                    img = self._combine_images(img, self._get_thread_image(last_nail, best_nail))
                    last_nail = best_nail
                    last_loss = best_loss
                    if self._debug:
                        print(f'Connection {len(nail_order)}: {best_nail} ({best_loss})')
                        cv2.imshow(W, img)
                        cv2.waitKey(1)
                    elif show_progress:
                        cv2.imshow(P, img)
                        k = cv2.waitKey(1)
                        if k == ord('q'): # quit
                            break

        self.nail_order = nail_order
        self.output_image = img
        if self._debug:
            cv2.imshow(W, self.output_image)
            cv2.waitKey(0)
        return nail_order, img

    def __find_best_nail(self, img: np.ndarray, last_nail: int, last_loss: float, pool: mp.Pool) -> tuple[int, float]:
        img_shared = np.ndarray(self.image_shape, dtype=np.float32, buffer=self.__mems[f'{SMP}img'].buf)
        img_shared[:] = img[:]
        losses = pool.starmap(self._combine_and_get_loss,
                              [(last_nail, nail) for nail in range(len(self._nails))],
                              chunksize=int(np.ceil(len(self._nails) / mp.cpu_count())))
        best_loss_idx = np.argmin(losses)
        best_loss = losses[best_loss_idx]
        if best_loss < last_loss:
            return best_loss_idx, best_loss
        return None, last_loss

    #def _combine_and_get_loss(self, *i: tuple[np.ndarray, int, int]) -> float:
        #img, last_nail, nail = i
    #def _combine_and_get_loss(self, img: np.ndarray, last_nail: int, nail: int) -> float:
    def _combine_and_get_loss(self, *i: tuple[int, int]) -> float:
        last_nail, nail = i
        if nail == last_nail:
            return np.Infinity # using the same nail twice is not allowed
        img = np.ndarray(self.image_shape, dtype=np.float32, buffer=self.__mems[f'{SMP}img'].buf)
        img = self._combine_images(img, self._get_thread_image(last_nail, nail))
        return self._loss(img)

    @abstractmethod
    def get_mask(self, image_shape: tuple[int]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_nails(self, n_nails: int) -> list[Point]:
        raise NotImplementedError

    def _create_thread_image(self, pos0: Point, pos1: Point, n0: int, n1: int, save: bool) -> np.ndarray|None:
        shift = 8
        img = np.ones((self.image_shape[0], self.image_shape[1]), dtype=np.float32)
        cv2.line(img,
                 (int(pos0[0] * 2**shift), int(pos0[1] * 2**shift)),
                 (int(pos1[0] * 2**shift), int(pos1[1] * 2**shift)),
                 0, # color
                 1, # thickness
                 cv2.LINE_AA,
                 shift)
        img = gaussian_filter(img, sigma=self._thread_width)
        img = img * self._thread_opacity + np.ones_like(img) * (1 - self._thread_opacity)
        img = (img * 255).astype(np.uint8)
        if save:
            mem = shared_memory.SharedMemory(create=False, name=f'{SMP}{n0}')
            img_shared = np.ndarray((len(self._nails) - n0, *self.image_shape), dtype=np.uint8, buffer=mem.buf)
            img_shared[n1-n0] = img[:]
            mem.close()
        else:
            return img

    def _combine_images(self, img: np.ndarray, thread_img: np.ndarray) -> np.ndarray:
        img = img.copy()
        img *= thread_img / 255.0
        return img

    def _loss(self, img: np.ndarray) -> float:
        return np.sum(np.abs(img[self._mask] - self.image[self._mask]))

    def _get_thread_image(self, n0: int, n1: int) -> np.ndarray:
        if n0 > n1:
            n0, n1 = n1, n0
        name = f'{SMP}{n0}'
        if name in self.__mems:
            mem = self.__mems[name]
            img_shared = np.ndarray((len(self._nails) - n0, *self.image_shape), dtype=np.uint8, buffer=mem.buf)
            return img_shared[n1-n0]
        else:
            img = self._create_thread_image(self._nails[n0], self._nails[n1], n0, n1, False)
            return img


class CircleThreadArtInfo(ThreadArtInfo):

    def get_mask(self, image_shape: tuple[int]) -> np.ndarray:
        # Create mask for circle
        assert image_shape[0] == image_shape[1], 'Image must be square'
        radius = min(image_shape) // 2
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.circle(mask, (radius, radius), radius, 1, -1)
        return mask.astype(np.bool_)


    def get_nails(self, n_nails: int) -> list[Point]:
        # Distribute nails evenly on a circle
        nails = []
        assert self.image_shape[0] == self.image_shape[1], 'Image must be square'
        radius = min(self.image_shape) // 2
        for i in range(n_nails):
            angle = i * 2 * np.pi / n_nails
            x = radius * np.cos(angle) + radius
            y = radius * np.sin(angle) + radius
            nails.append((x, y))
        return nails



class RectangleThreadArtInfo(ThreadArtInfo):

    def get_mask(self, image_shape: tuple[int]) -> np.ndarray:
        return np.ones(image_shape, dtype=np.bool_)

    def get_nails(self, n_nails: int) -> list[Point]:
        # Distribute nails nearly evenly on a rectangle, the corners should be exact
        w = self.image_shape[1]
        h = self.image_shape[0]
        # Keep the aspect ratio of the image
        a = w / (2 * (w + h)) # a in [0, 0.5]
        w_nails = int(n_nails * a)
        h_nails = int(n_nails * (0.5 - a))
        if w > h:
            w_nails += 1
        else:
            h_nails += 1
        nails = []
        for x in np.linspace(0, w, w_nails-1, endpoint=True):
            nails.append((x, 0))
        for y in np.linspace(0, h, h_nails-1, endpoint=True):
            nails.append((x, h-1))
        for x in np.linspace(w, 0, w_nails-1, endpoint=True)[::-1]:
            nails.append((w-1, y))
        for y in np.linspace(h, 0, h_nails-1, endpoint=True)[::-1]:
            nails.append((0, y))

        return nails



def main(options: argparse.Namespace|object) -> ThreadArtInfo:
    # Load image and convert to grayscale
    image = np.array(Image.open(options.image).convert('L'))
    assert image is not None, 'Image not found'
    assert image.ndim == 2, 'Image must be grayscale'
    if options.debug:
        print(image.shape)
        cv2.namedWindow(W, cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(W, image)
        cv2.waitKey(0)

    # Clip image if the shape is a circle
    if options.shape == 'circle':
        radius = min(image.shape) // 2
        image = image[image.shape[0]//2-radius:image.shape[0]//2+radius, image.shape[1]//2-radius:image.shape[1]//2+radius]

    # Resize image
    w = options.working_size
    h = int(image.shape[0] * w / image.shape[1])
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    if options.debug:
        print(image.shape)
        cv2.imshow(W, image)
        cv2.waitKey(0)

    # Create ThreadArtInfo object
    if options.shape == 'circle':
        tai = CircleThreadArtInfo(options.nails, options.debug)
    elif options.shape == 'rectangle':
        tai = RectangleThreadArtInfo(options.nails, options.debug)
    else:
        raise ValueError('Unknown shape')

    tai.prepare(image.shape, options.thread_width, options.thread_opacity, options.load_nails is None)

    if options.load_nails is not None:
        tai.load(options.load_nails)
        if options.to_video is None:
            tai.apply_loaded_sequence(options.visualize_progress)
        else:
            video = cv2.VideoWriter(options.to_video,
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    options.video_fps,
                                    image.shape,
                                    False)
            if options.video_length is not None:
                n_frames = int(options.video_length * options.video_fps)
                frames_to_generate = np.linspace(0, len(tai.nail_order) - 1, n_frames, endpoint=True, dtype=np.int32)
            else:
                frames_to_generate = None
            for i, frame in enumerate(tai.generate_frames(False)):
                if frames_to_generate is not None and i not in frames_to_generate:
                    continue
                video.write((frame * 255).astype(np.uint8))
            if options.video_blend_target > 0:
                n_frames = int(options.video_blend_target * options.video_fps)
                last_frame = tai.output_image
                target_frame = (image / 255.0) if image.dtype == np.uint8 else image
                for i in trange(n_frames):
                    alpha = np.cos(i / (n_frames - 1) * 2 * np.pi) / 2 + 0.5
                    assert alpha >= 0 and alpha <= 1, 'Alpha out of range'
                    out_frame = last_frame.copy()
                    out_frame[tai._mask] = last_frame[tai._mask] * alpha + target_frame[tai._mask] * (1 - alpha)
                    video.write((out_frame * 255).astype(np.uint8))
                for i in trange(int(options.video_end * options.video_fps)):
                    video.write((last_frame * 255).astype(np.uint8))

            video.release()
    else:
        tai.calculate(image, show_progress=options.visualize_progress, early_end=not options.retry_on_dead_end)

    if options.debug:
        cv2.destroyAllWindows()

    return tai


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('image', nargs='?', help='Image to convert', type=str, default='../demo/pexels-wendel-moretti-1925630.jpg')
    p.add_argument('shape', nargs='?', help='Shape of the final image', choices=['circle', 'rectangle'], default='circle')
    p.add_argument('nails', nargs='?', help='Number of nails', type=int, default=200)
    p.add_argument('--working-size', help='Width of the internal image', type=int, default=1250)
    p.add_argument('--thread-width', help='Width of the thread in relation to width of the internal image', type=float, default=0.001)
    p.add_argument('--thread-opacity', help='Opacity of the thread', type=float, default=0.2)
    p.add_argument('--visualize-progress', help='Show progress during generation', action='store_true')
    p.add_argument('--retry-on-dead-end', help='Try to find a new nail pair if no nail can be found', action='store_true')
    p.add_argument('--load-nails', help='Load nails from file', type=str, default=None)
    p.add_argument('--to-video', help='Save result as video', type=str, default=None)
    p.add_argument('--video-fps', help='FPS of the video', type=float, default=60.0)
    p.add_argument('--video-length', help='Length of the video in seconds', type=float, default=None) # None = all frames once
    p.add_argument('--video-blend-target', help='At the end of the video, blend the last frame with the target image for x seconds', type=float, default=0.0)
    p.add_argument('--video-end', help='At the end of the video show the final image for x seconds', type=float, default=0.0)
    p.add_argument('--debug', help='Show debug information', action='store_true')

    options = p.parse_args()

    if options.debug:
        print(options)
    with main(options) as tai:
        if options.load_nails is None:
            out_filename_base = '.'.join(options.image.split('.')[:-1])
            np.savetxt(f'{out_filename_base}-nails.txt', np.array(tai.nail_order), fmt='%d')
            cv2.imwrite(f'{out_filename_base}-result.png', (tai.output_image * 255).astype(np.uint8)) # PNG encoder needs [0, 255] range
        # show result
        cv2.namedWindow('Result', cv2.WINDOW_GUI_NORMAL)
        cv2.imshow('Result', tai.output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
