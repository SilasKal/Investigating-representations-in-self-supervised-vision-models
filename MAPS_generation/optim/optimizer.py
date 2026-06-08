import time
from pathlib import Path

import numpy as np


class SceneOptimizer:
    def __init__(self, scene, param_space, evaluator):
        self.scene = scene
        self.ps = param_space
        self.evaluator = evaluator

    def run(
        self,
        search_optimizer,
        max_iter=100,
        stop_prob=None,
        verbose=True,
        path_render=None,
        logger=None,
    ):
        best = {
            'x': None,
            'theta': None,
            'score': -np.inf,
            'prob': None,
            'iter': None,
        }

        history = []
        object_name = self.scene.object_name
        if path_render is None:
            path_render = Path('tmp') / object_name
            path_render.mkdir(parents=True, exist_ok=True)
        digits = len(str(max_iter))

        for it in range(max_iter):
            iter_start = time.perf_counter()
            path_render_iter = path_render / f'iter_{it:03d}'
            path_render_iter.mkdir(parents=True, exist_ok=True)

            # Sample candidates
            X = search_optimizer.ask()

            images = []
            for i_x, x in enumerate(X):
                theta = self.ps.decode(x)
                self.scene.set(theta)
                img_path = path_render_iter / f'candidate_{i_x:02d}.png'
                images.append(self.scene.render(img_path))

            # Find best candidate
            scores, probs = self.evaluator.evaluate(images)
            iter_best_idx = np.argmax(scores)
            iter_best_score = scores[iter_best_idx]
            iter_mean_score = np.mean(scores)
            iter_best_prob = probs[iter_best_idx, self.evaluator.class_id]
            iter_predicted_class = probs[iter_best_idx].argmax()

            if scores[iter_best_idx] > best['score']:
                best.update({
                    'x': X[iter_best_idx].copy(),
                    'theta': self.ps.decode(X[iter_best_idx]),
                    'score': iter_best_score,
                    'prob': iter_best_prob,
                    'pred': iter_predicted_class,
                    'iter': it,
                    'idx': iter_best_idx,
                })

            history.append({
                'iter': it,
                'best_x': best['x'],
                'best_theta': best['theta'],
                'best_score': best['score'],
                'best_prob': best['prob'],
                'best_pred': best['pred'],
                'iter_mean_score': iter_mean_score,
                'iter_best_score': iter_best_score,
                'iter_best_prob': iter_best_prob,
                'iter_predicted_class': iter_predicted_class,
            })

            # Minimize
            search_optimizer.tell(X, -scores)

            iter_time = time.perf_counter() - iter_start

            if verbose:
                if logger:
                    logger.info(
                        f"[{it+1:{digits}d}/{max_iter:{digits}d}] "
                        f"Best score: {best['score']:7.3f} | "
                        f"Best prob: {best['prob']:.3f} ({best['pred']}) | "
                        f"Iter. mean score: {iter_mean_score:7.3f} | "
                        f"Time/iter.: {iter_time:.2f}s"
                    )
                else:
                    print(
                        f"[{it+1:{digits}d}/{max_iter:{digits}d}] "
                        f"Best score: {best['score']:7.3f} | "
                        f"Best prob: {best['prob']:.3f} ({best['pred']}) | "
                        f"Iter. mean score: {iter_mean_score:7.3f} | "
                        f"Time/iter.: {iter_time:.2f}s"
                )

            if stop_prob is not None and best['prob'] >= stop_prob:
                break

        return best, history
